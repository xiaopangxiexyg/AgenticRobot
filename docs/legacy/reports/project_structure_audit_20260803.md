# DexTeleop 项目结构只读审计历史快照（2026-08-03）

审计日期：2026-08-03
历史状态：问题定位完成；本文不再作为当前实施计划
范围：当前 `/home/zc/HandMotion/Developing/DexTeleop` 工作区；以工作区实际内容为准，不把 `HEAD` 当作当前运行行为

当前结构治理以
[`todos/dexteleop_p0_structure_remediation_plan_20260803.md`](../../../todos/dexteleop_p0_structure_remediation_plan_20260803.md)
和项目 [`AGENTS.md`](../../../AGENTS.md) 为准。本文保留审计时点的规模、问题与证据，
不得把其中的行数、测试数量或并行状态解释为当前事实。

## 1. 审计目标与边界

本轮只定位以下问题，并给出后续整理顺序：

- 巨型文件、巨型函数、巨型类；
- 一个包或文件混合多个 owner、状态机或资源生命周期；
- 新旧包并存、兼容层长期化、重复实现；
- 扁平目录、过宽包级导出和难以定位的公共接口；
- runtime、vendor、实验工具、证据文件和静态资源混放；
- 测试和文档结构不能准确反映当前生产结构。

本轮没有：

- 修改 Python、配置、依赖或锁文件；
- 启动设备、相机、网络服务、仿真 viewer 或真实运动；
- 清理工作区、移动文件、删除兼容入口或重写公共 API；
- 把行数本身当作缺陷。文件只有在同时出现职责聚集、复杂度、依赖或生命周期问题时才列为重构对象。

审计开始时工作区已有并行改动，集中在 `control_composition/trajectory/`、
`robot_motion/action/trajectory.py`、`robot_motion/deploy/trajectory_adapter.py`、相关测试和规划文档。
审计进行期间又出现了 `robot_motion/position_control/horizon_governor.py` 及其测试修改，说明工作区仍在
并发演进。后续任何实现都必须在开始前重新读取 status/diff，并避开或重新审查这些并发修改。

## 2. 结论先行

当前结构比 2026-07 中旬明显进步：真机 canonical 入口
`teleop_runtime/debug/run_vive_rm65_real_teleop.py` 已从历史约 1.7～1.9 万行降到
1,758 行，`collection_runtime`、`control_composition`、`control_station`、
`robot_motion/action` 和 `robot_motion/deploy` 的 owner 也已经建立。

但项目尚未完成结构收口，主要问题已转移为：

1. **一个现存的包边界回归。** `teleop_runtime` 新增了两个未登记的
   `control_station` 反向依赖，当前边界测试实际为 `1 failed, 47 passed`。
2. **多个权威对象仍以单类或单函数承载整个子系统。** 最突出的是 10,799 行的
   双臂 target solver、3,213 行的 ControlTrace completeness 函数、3,265 行的真机
   control loop 和 2,751 行的 policy deploy `main()`。
3. **真机入口虽然变薄，旧入口仍是隐式依赖容器。** real-teleop 子模块通过
   205、96、92 个名字的动态 support 注入维持旧 monkeypatch/import 语义；这是
   “文件搬走了，但依赖图尚未真正拆开”。
4. **新旧 ownership 双轨运行。** `robot_motion/position_control` 与
   `teleop_runtime/position_control`、`collection_runtime` 与
   `teleop_runtime/collection`、`control_station` 与 real-teleop GUI/progress
   仍有迁移接缝；其中有些是合理的 frontend adapter，有些已经跨越包边界。
5. **公共包过度扁平且根导出过宽。** `dataset_contracts`、`dataset_interface`、
   `robot_motion/action`、`robot_motion/deploy`、`teleop_runtime/real_teleop` 和
   `teleop_runtime/collection` 都有几十个同级模块；几个 `__init__.py` 已成为数百个符号的
   eager facade。
6. **工具、静态资源、vendor 和现场证据混放。** 通用名 `utils` 已经造成真实的
   vendor `utils.py` shadowing 防御代码；Web UI 把约 5,900 行 HTML/CSS/JS 嵌在 Python；
   `hardware/rm65/utils/results` 与 runtime package 同树；整个 `assets/` 被忽略但运行时大量引用。
7. **测试和临时文档落后于当前结构。** 448 个测试 Python 文件全部平铺在
   `tests/` 根目录，最大的测试文件 6,286 行；两个根目录 `*_TMP.md` 仍描述已经过时的
   1.4～1.8 万行入口快照。

因此，推荐顺序不是继续机械拆文件，而是先恢复边界、收敛兼容注入，再按 owner 拆巨型对象；
最后才整理公共 facade、测试树和资产/证据目录。

## 3. 当前规模快照

以下统计排除了 `legacy/`、`build/`、`assets/`、`remote_data/`、`teleop_records/`、
根 `tests/`、`todos/` 和明确的 RM65 vendor SDK 源码，保留项目维护的 runtime、工具和 hardware adapter。

| 包 | Python 文件 | 行数 | ≥1000 行文件 | ≥2000 行文件 | 最大文件 |
|---|---:|---:|---:|---:|---|
| `teleop_runtime` | 361 | 172,474 | 41 | 19 | `deploy/run_policy.py` 6,835 |
| `robot_motion` | 173 | 94,438 | 19 | 6 | `rm65/dual_arm_target_solver.py` 10,799 |
| `hardware` | 172 | 65,670 | 16 | 3 | `rm65/impedance.py` 5,300 |
| `utils` | 110 | 48,538 | 14 | 2 | `portable_vis/.../dmtac_viewer.py` 3,018 |
| `control_station` | 60 | 45,529 | 10 | 5 | `web/calibration_web_ui.py` 7,216 |
| `sim` | 144 | 42,628 | 6 | 1 | `mujoco/deploy/runtime.py` 3,346 |
| `dataset_contracts` | 50 | 28,448 | 9 | 1 | `completeness.py` 4,075 |
| `dataset_interface` | 62 | 20,909 | 2 | 0 | `composition_control_trace.py` 1,629 |
| `control_composition` | 38 | 20,169 | 5 | 1 | `composition/core.py` 2,363 |
| `deployments` | 40 | 12,040 | 1 | 0 | `openpi/client/rtc.py` 1,047 |
| `collection_runtime` | 35 | 8,471 | 0 | 0 | `component_capture/episode_store.py` 980 |

需要特别区分：RM65 官方 SDK 的 `rm_robot_interface.py`、`rm_ctypes_wrap.py` 和
`robotic_arm.py` 虽然也很大，但它们是 vendor snapshot，不应与第一方 giant owner 用同一
重构策略。首要工作是隔离 vendor 边界和 provenance，而不是改写其内部实现。

### 3.1 最拥挤的目录

| 目录 | Python 文件 | 行数 | 结构信号 |
|---|---:|---:|---|
| `teleop_runtime/real_teleop` | 75 | 42,385 | 已拆出很多模块，但 coordinator/loop/runtime 仍靠动态全局依赖连接 |
| `teleop_runtime/collection` | 63 | 27,979 | source-bound frontend、通用 writer、兼容 shim 和旧 collector 并存 |
| `dataset_contracts/` 根 | 50 | 28,448 | ControlTrace、progress、版本 schema、model continuation 全部平铺 |
| `robot_motion/action` | 43 | 23,723 | action、authority、safety、execution、reference、trajectory 同层 |
| `robot_motion/deploy` | 34 | 20,373 | protocol、worker、runtime、RTC、observation、codec 同层 |
| `dataset_interface/` 根 | 54 | 19,319 | progress 系列约二十个文件与 dataset/video/model 接口平铺 |
| `control_station` 根 | 19 | 15,687 | core、Sim-HIL、acceptance、GUI bootstrap 混在根层 |

## 4. 已确认的问题

### P0-1：`control_station` application-leaf 边界当前已回归

AGENTS.md 要求下层包不得导入 `control_station`。当前已有测试允许一组明确的临时
application-edge seam，但实际代码比 allowlist 多出两处：

- `teleop_runtime/real_teleop/dataset_recorder.py`
- `teleop_runtime/real_teleop/progress_arm_control.py`

只读执行：

```bash
/home/zc/anaconda3/envs/dexteleop310/bin/python -m pytest -q \
  tests/test_control_station_boundaries.py \
  tests/test_collection_runtime_boundaries.py \
  tests/test_control_composition_boundaries.py \
  tests/test_dataset_dependency_boundaries.py
```

结果：`1 failed, 47 passed`。失败不是环境或硬件问题，而是 AST 边界断言准确发现这两个
额外反向依赖。

根因不是简单“少更新了 allowlist”：

- `control_station.real_teleop.progress` 声明自己拥有 UI-facing operator intent 和
  availability；
- `dataset_recorder` 和 `progress_arm_control` 属于下层 entity/runtime；
- 它们直接使用 Station 的 mode 常量、normalizer 和 `ProgressOperatorIntent`，使应用叶
  变成下层 domain type provider。

推荐方向是保留 Station 对 operator intent 的所有权，但由下层拥有一个设备/UI 无关的
typed input port 或 domain command contract，Station adapter 做显式投影。不要把两处文件
简单加入 allowlist；测试注释已明确禁止 device/control/safety implementation 扩大豁免。

### P0-2：双臂 target solver 已成为 9,716 行的单一巨型类

`robot_motion/rm65/dual_arm_target_solver.py`：

- 文件 10,799 行；
- `DualArmCollisionAwareRetargeter` 类 9,716 行；
- `_solve_bimanual_core()` 1,168 行；
- `_select_safe_tracking_candidate()` 915 行；
- `_dual_arm_singularity_escape_candidates()` 725 行；
- `_evaluate_dual_arm_step_candidate()` 827 行。

该类同时包含：

- target/context/command conversion；
- 双臂 IK 求解；
- 碰撞检查和候选评估；
- output qpos smoothing；
- wrist q5 hysteresis、branch guard 和 continuity；
- singularity escape/posture escape/wrist unload；
- joint-limit boundary hold、escape latch 和方向状态；
- inactive-side execution cap；
- jump gate、hold/recovery；
- profiling、warning 和大批 metadata 生成。

这里确实需要拆分，但必须保留一个 canonical retarget/solve owner。推荐把纯 contract、候选
生成、候选评估、各类 recovery policy、smoothing/hysteresis state 和 diagnostics 拆成由
同一个 facade 实例私有持有的子组件；不能产生第二个 IK owner、第二个 safety decision 或
另一个可独立提交 q_ref 的 runtime。

### P0-3：真机 real-teleop 的“物理拆分”尚未变成“依赖拆分”

当前动态 support binding 规模：

| 模块 | 注入名字数量 |
|---|---:|
| `teleop_runtime/real_teleop/runtime.py` | 205 |
| `operator_control_runtime.py` | 96 |
| `control_loop.py` | 92 |
| `cli_resolution.py` | 38 |
| `debug_log_frames.py` | 24 |
| `cli_runtime_config.py` | 23 |
| `startup_runtime.py` | 20 |

canonical debug 入口在 import 后多次把 `globals()` 注入上述模块。GUI 兼容模块甚至通过
`sys.modules[__name__] = _implementation` 保持旧 monkeypatch target 的同一对象身份。

这套机制在迁移早期有价值，但现在造成：

- 依赖只能从字符串白名单和运行时 global lookup 理解；
- 模块签名不能表达真正需要的 capability；
- 单测大量冻结旧入口私有名字与 monkeypatch 行为；
- 新模块仍可隐式获得硬件、writer、GUI 和安全副作用；
- `runtime.py`、`control_loop.py` 和 GUI coordinator 无法独立构造或局部验证。

当前最大的三个 remaining god units 是：

- `RealTeleopControlInput`：5,193 行、约 180 个方法，混合 UI 状态、foot-pad、init/fix arm、
  tracker-side gate、calibration、episode、gripper force、progress 和状态发布；
- `run_real_teleop_control_loop()`：3,265 行，约 458 个分支节点，混合 control tick、
  progress arm plan、recording、force collision、Hold、debug、dispatch fault 和 UI publish；
- `run_real_teleop()`：2,178 行，负责资源组装、启动、闭包式回调和 loop handoff。

推荐先用少量 typed dependency bundles/ports 替换 globals 注入，再拆内部状态机；继续逐个搬 helper
只会保留厚兼容壳。

### P0-4：ControlTrace completeness 是一个 3,213 行单函数

`dataset_contracts/completeness.py:evaluate_control_trace_lineage()`：

- 3,213 行；
- 约 528 个分支节点；
- 约 950 个调用节点；
- 只有一个最终 return。

这说明所有 event family 的索引、依赖、identity、顺序和 issue 生成被串成一个程序块。
它位于纯 contract 包，行为又直接决定 episode 可提交性，因此修改风险很高。

推荐按稳定 event/domain family 拆成确定顺序的纯 validator，例如 session/activation、proposal/
authority、safety/reference/dispatch、composition、native stream、progress 等。顶层函数仍是唯一
公开入口，保持完全相同的 issue code、顺序、profile 语义和 canonical hash 输入；不引入动态
plugin discovery，也不让 validator 访问 runtime 或 I/O。

### P1-1：policy deploy frontend 同时是 CLI、factory、runtime 和控制循环

`teleop_runtime/deploy/run_policy.py` 6,835 行：

- `add_args()` 964 行；
- `validate_args()` 488 行、约 161 个分支；
- `main()` 2,751 行、约 423 个分支；
- 文件还拥有 debug logger、arm state reader、camera/source/backend factory、Hold/ESTOP helper、
  sync/RTC profile 分流、observation、inference、queue consumption、canonical preparation、
  position executor、dispatch 和 cleanup。

该文件是 canonical deploy frontend，可以保留薄入口身份，但当前不应继续吸收 RTC、模型 adapter
或 hardware profile 的新分支。后续应形成：CLI schema/resolve → resource transaction → policy
producer/coordinator → execution session → single cleanup owner。

### P1-2：headless collector 仍是 6,213 行兼容主程序

`teleop_runtime/collection/run_collect.py` 包含：

- 1,192 行 `parse_args()`；
- source/backend/retargeter factory；
- runtime Home；
- recovery/return-home 状态；
- replay 与 unrecorded replay；
- episode finalize；
- terminal state；
- status/debug；
- 466 行 `main()`。

该入口按 AGENTS.md 必须作为 headless/compatibility canonical entrypoint 保留，不能直接归档。
问题在于它仍是旧 collector 的行为 owner，而不是薄 frontend。应先明确哪些是 source-bound
teleop adapter，哪些通用 writer/episode mechanics 已归 `collection_runtime`，再逐步下沉；不能把
整个目录机械搬到 `collection_runtime`。

### P1-3：deploy 与 collect 已出现可证实的配置/构造重复

`run_policy.py` 与 `run_collect.py` 有 17 个同名顶层定义。高相似或完全相同的实现包括：

- `_hand_backend_mode()`：1.000；
- `_load_gripper_task_profile()`：1.000；
- `_parse_dmtac_tactile_fields()`：1.000；
- `_profile_value()`：1.000；
- `_dmtac_backend_configs()`：0.994；
- `_positive_float()`：0.967；
- `_dmtac_tactile_enabled()`：0.910；
- `_parse_arm_max_angvel()`：0.853。

这不是把两个 application loop 合并的理由；它说明 DMTAC/gripper profile、CLI value parsing 和
backend config resolution 缺少真实 owner。推荐把纯 profile resolution 放到
`teleop_runtime/configs` 或对应 gripper adapter owner，两个 frontend 只消费解析结果。

### P1-4：position execution 的共享语义与真机 adapter 分层不完整

当前同时存在：

- `robot_motion/position_control/`：14 个文件、6,804 行，拥有 shared profile、resampling、
  trajectory、safety、capability 和 horizon；
- `teleop_runtime/position_control/`：10 个文件、9,363 行，最大
  `deploy_executor.py` 4,278 行。

`PositionDeployExecutor` 单类 3,877 行，包含 keyframe admission、canonical safety、collision reset、
reference generation、execution cancellation、background worker、discrete dispatch、ControlTrace observer、
fault latch 和 backend callback。

并非整类都应迁到 `robot_motion`：真实 RM65 dispatch callback 和 application observation adapter
属于 teleop frontend。但 reference scheduling、epoch/cancellation、generic admission 和 execution work
语义属于 shared motion。应先画清对象和 callback 的 owner，再移动纯 shared core，保留一个真机
binding；不能产生第二个 queue、reference publisher 或 dispatch owner。

### P1-5：RM65 backend 和 impedance controller 混入过多 shared motion 语义

`teleop_runtime/backends/arms/rm65.py`：

- 文件 5,682 行；
- `RM65ArmBackend` 类 4,943 行；
- `send_command()` 726 行；
- background stream loop、jump gate、singularity protection、SDK I/O、state refresh、trace 和 lifecycle
  集中在同一类。

`hardware/rm65/impedance.py`：

- 文件 5,300 行；
- `RM65BaseImpedanceController` 类 3,747 行；
- `step()` 789 行；
- 同时拥有 vendor calls、admittance、IK resolver、joint limit/rate guard、failure hold、dual-rate IK、
  follow stream、debug metadata 和 lifecycle。

硬件包可以拥有 vendor device lifecycle 和 backend-specific fail-closed guard，但 shared IK selection、
joint motion limiting、reference timing和安全 contract 应尽量由 `robot_motion` 提供。推荐形成“纯 motion
policy/core + RM65 transport/device adapter”，而不是简单把整个 impedance 文件搬走。

### P1-6：包级 facade 过宽且 eager import 成本很高

当前运行时 `__all__` 规模：

| facade | 导出符号数 |
|---|---:|
| `dataset_contracts` | 549 |
| `robot_motion.action` | 337 |
| `robot_motion.deploy` | 266 |
| `dataset_interface` | 233 |
| `teleop_runtime` | 72 |
| `control_composition` | 63 |
| `control_station` | 60 |

当前环境中的只读 import probe 显示：

- `import dataset_contracts` 加载约 409 个模块、48 个自身子模块；
- `import dataset_interface` 加载约 612 个模块、约 55 个自身子模块；
- `import control_composition` 加载约 571 个模块；
- `import control_station` 加载约 659 个模块；
- `import teleop_runtime` 加载约 786 个模块，并拉入 hardware、multiprocessing、socket、NumPy、SciPy、OSQP；
- 相比之下 `collection_runtime`、`deployments` 和 `sim` 根包只加载自身。

这不等于 import 会立即启动硬件，但它扩大了循环依赖、worker spawn、可选依赖和包内任意改动的
影响面。历史上 `dataset_interface` aggregate import 已经造成过 worker 环境加载 OSQP 的问题。

不要一次性删根导出。应先用静态 call-site 清单区分 supported public API、repo-internal convenience
和 legacy identity，推动新代码使用 capability-named submodule；之后再用显式 deprecation 或 lazy
attribute 逐批收窄。

### P1-7：兼容 shim 数量大且分散，迁移状态难以阅读

静态检查在 `teleop_runtime` 中识别出 74 个不超过 100 行的 import/re-export candidate，其中包括
package facade，也包括大量明确兼容路径：

- `teleop_runtime.collection.* → collection_runtime.*`；
- `teleop_runtime.action.* → robot_motion.action.*`；
- `teleop_runtime.deploy.* → robot_motion.deploy.*`；
- `teleop_runtime.rigs/interfaces → robot_motion.runtime`；
- `teleop_runtime.gui.* → control_station.*`；
- `teleop_runtime.real_teleop.dataset_layout → collection_runtime.dataset_layout`。

兼容 shim 本身不是错误，但当前它们散落在正常实现目录中，缺少统一 registry：看目录无法判断某文件
是 owner、adapter 还是待删除 alias。建议建立一份机器可检验的 compatibility manifest，记录 old path、
canonical path、public/内部级别、调用者和删除 gate；不建议新建一个承载实现的 `assembly/` 或
`compatibility` 杂物包。

### P1-8：`utils` 是真实发生过命名冲突的混合包

`utils` 目前 110 个 Python 文件、48,538 行，混合：

- dataset postprocess；
- hand-eye/head-eye/UMI calibration；
- visualization GUI；
- statistics/report generation；
- random-background augmentation；
- wrist mask tools。

它还被列入 production package discovery。canonical real-teleop 入口必须在 import 前执行
`_remove_dmrobotics_package_path_shadow()`，专门移除 vendor SDK 中的 `dmrobotics/utils.py` 路径，证明
通用顶级名字已造成实际冲突。

此外，`dataset_interface` 直接依赖 `utils.random_background`，让数据接口依赖一个通用工具包；
`hardware/h1-2` 又依赖 cwd-local `utils.py`，其 `from utils import *` 在仓库模块上下文会与顶级
`utils` 冲突。

推荐按 owner 拆出明确包名：数据增强/掩码的可复用实现归 dataset interface 或明确的 data transform
package；calibration/visualization/report CLI 归 `tools/` 或领域命名包；H1-2 脚本改成明确本地模块导入。
在调用者迁完前保留窄 shim，最后才考虑移除顶级 `utils`。

### P2-1：Web UI 的 Python 文件实际包含约 5,900 行前端源码

`control_station/web/calibration_web_ui.py` 7,216 行，但 Python 定义从约 5,949 行才开始。前面是一个
巨大 `_HTML_PAGE`，内含 HTML、CSS 和多段 JavaScript；同时目录里又有
`real_teleop_progress.js`、`operator_module_layout.js`、`sidebar_resize.js`、`input_activity.js` 和
`static/` 资源。

这是高收益、低控制语义风险的整理点：把 HTML/CSS/JS 拆成 package-data/static template，Python
只保留 state、event encoding、HTTP lifecycle 和模板参数。必须保持单文件安装后的 package-data
可发现性、fallback 脚本、页面版本和 UI 契约测试。

### P2-2：Control Station 的 Sim-HIL 应用仍过厚

`control_station/sim_hil.py` 4,970 行，包含 config、profile/anchor binding、readiness provider、source
construction、status projection、application lifecycle、CLI、dry-run 和 acceptance GUI。
`MujocoSimHILApplication` 类约 1,983 行，`status()` 509 行。

它属于 concrete application profile，导入 sim 是允许的，但可以按 application/config/source binding/
status/CLI 分成 `control_station/sim_hil/` 子包。拆分时 application 仍是唯一 lifecycle owner，不能把
sim execution、authority或 recorder 复制进 Station。

### P2-3：测试目录和测试文件同样失去结构

当前：

- `tests/` 根目录有 448 个 Python 文件；
- 没有任何嵌套 Python test package；
- production 有 11 个顶级包，但测试不按 owner 镜像；
- 最大测试文件 `test_real_teleop_gui_collection_visualization.py` 6,286 行；
- 其次 `test_dual_arm_collision_aware_safety.py` 4,751 行、
  `test_control_station_mujoco_pose_human_replay.py` 3,567 行、
  `test_rm65_impedance.py` 3,464 行。

建议在生产边界稳定后按 `tests/unit/<owner>/`、`tests/contract/`、`tests/integration/<profile>/` 和
明确的 hardware/manual 层次组织。先保持 marker 和命令兼容，避免单纯移动测试导致现有脚本失效。
巨型测试按行为 contract 拆，不按被测函数行号拆。

### P2-4：runtime、vendor、现场证据和生成结果混放

已确认的结构信号：

- `hardware/rm65/utils/results/` 有 29 个 tracked result/data 文件，最大的约 2.26 MB；
- `utils/statistic/*_reports/` 保存 tracked 生成报告；
- `teleop_records/` 跟踪 70 个标定/profile 文件，但同一目录名又用于本地 runtime output；
- `hardware/pyzlg_dexhand` 是带独立 `pyproject.toml`、`setup.py`、二进制库和 examples 的嵌套项目；
- RM65 SDK 源码和 `.so` 与第一方 adapter 同在 `hardware/rm65`；
- 当前工作区 `assets/` 约 3.1 GB，整个目录被 `.gitignore` 忽略，但 sim、robot_motion、安装脚本和文档
  大量引用其中 URDF、mesh 和 DM-Tac SDK。

这些文件不能直接清理或移动。首先需要分类 manifest：

- immutable runtime contract asset；
- vendor SDK snapshot；
- checked-in test fixture/evidence；
- operator-specific calibration；
- generated report；
- local cache/log/raw data。

然后为外部 asset 建立版本、hash、获取方式和缺失时 fail-closed 的 provenance。没有独立备份、调用者
清单和 hash receipt 前，不进行任何批量移动或删除。

### P2-5：临时架构文档已经失真

根目录仍有：

- `REAL_TELEOP_REFACTOR_TMP.md`：审查日期 2026-07-15，描述入口约 14,536 行；
- `REAL_TELEOP_CONTROL_CHAIN_AUDIT_TMP.md`：快照日期 2026-07-17，行号和若干实现边界已过时。

当前入口实际 1,758 行。这两份文档本身写明“重构完成后归档或删除”，但尚未更新，也没有迁移状态
索引。它们容易被误当作当前事实源。应先核对仍有效的不变量和未完成 gate，将其迁入当前架构文档或
本审计计划，再按 legacy policy 检查所有引用后归档；不能直接删除。

## 5. 事实与待验证假设

### 已确认事实

- 上述文件行数、定义跨度、support 名字数量、根导出数量和 import probe 来自当前工作区；
- 四组边界测试实际得到 `1 failed, 47 passed`；
- deploy/collect 的同名 helper 存在高相似或完全相同实现；
- `calibration_web_ui.py` 的前约 5,900 行是嵌入式前端资源；
- `utils` shadowing 已被 canonical 入口中的显式 bootstrap 防御记录；
- `assets/` 被忽略但当前代码和文档大量引用；
- 当前并行 trajectory 改动未被本轮修改。

### 仍需在实施前验证

- 每个动态 support 名字是否仍有 repo 外部调用者或只是旧测试 monkeypatch 点；
- `teleop_runtime.position_control` 中哪些对象能完全下沉，哪些必须保留 real backend adapter；
- impedance shared policy 的数值行为是否已被其他真机脚本直接依赖；
- `run_collect` 的每个 helper 是否还有外部程序化 import；
- tracked 现场 result/profile 哪些是正式可复现 evidence，哪些只是历史生成物；
- ignored assets 的正式来源、许可证、版本和远端同步机制；
- TMP 文档是否仍被外部文档、脚本或团队流程引用。

## 6. 推荐实施顺序

以下阶段都需要用户确认后再开始。每阶段只处理一个 ownership 边界，不与当前 trajectory 主线混批。

### 阶段 0：冻结行为与恢复边界

目标：在不做大规模搬迁前，建立可信基线。

1. 解决 `dataset_recorder.py`、`progress_arm_control.py` 对 Station 的反向依赖；推荐引入下层 neutral
   port/command contract，由 Station adapter 投影，不扩大 allowlist。
2. 生成 compatibility manifest：旧路径、canonical owner、是否 public、repo 内调用者、repo 外兼容风险、
   删除条件。
3. 为 canonical 三入口保存 `--help`/参数解析快照和 import-side-effect 测试。
4. 冻结真机关键行为：single/dual、position/impedance、frame、command type、Hold/Home、filter 顺序、
   startup rollback 和 reverse cleanup。
5. 为 giant owner 建立 characterization fixtures；不运行硬件。

完成 gate：全部 package boundary tests 恢复通过，且没有新增 allowlist 或第二 authority/queue/dispatcher。

### 阶段 1：低控制风险、高收益整理

1. 把 `calibration_web_ui.py` 的 HTML/CSS/JS 移到 package-data/static template；保持 HTTP/UI contract。
2. 拆 `dataset_contracts/completeness.py` 为确定顺序的纯 validator modules；顶层 API 和 issue 顺序不变。
3. 建立 asset/vendor/evidence 分类清单和 hash/provenance 文档，但暂不移动大文件。
4. 将仍有效的 TMP 内容并入当前架构索引；满足 legacy policy 后再归档旧文档。

### 阶段 2：收敛 real-teleop 兼容桥

1. 用 `RealTeleopRuntimeDependencies` 或若干小型 capability ports 替代 205-name globals 注入。
2. 将 `RealTeleopControlInput` 分成：
   - pure operator state/reducer；
   - foot-pad/input routing；
   - runtime Home/tracker-side operation coordinator；
   - init/fix arm operation adapters；
   - episode/progress/gripper operation adapters；
   - rebuildable status projection。
3. 将 3,265 行 control loop 按固定 stage 拆成可测试组件，但保留一个 loop owner 和一个 dispatch path。
4. 旧 debug entrypoint 只保留 canonical CLI 和明确登记的兼容导出；迁移期允许一层 adapter，不再让新模块
   绑定完整 `globals()`。

### 阶段 3：拆分 shared motion giant owners

1. `DualArmCollisionAwareRetargeter` 保留 facade/authoritative state；内部按 candidate generation、evaluation、
   singularity recovery、joint-limit recovery、smoothing/hysteresis、diagnostics 拆为私有 owned components。
2. 把 `PositionDeployExecutor` 的 generic reference/admission/cancellation core 下沉到 `robot_motion`；
   real observation/dispatch binding 留在 `teleop_runtime`。
3. 把 RM65 position/impedance 的 vendor I/O 与 shared IK/rate/safety policy 分层；保持 backend 唯一 lifecycle
   owner 和 fail-closed stop。
4. 每一步用 offline replay/fake backend 做等价性验证；不把离线通过写成现场验证。

### 阶段 4：瘦身 canonical frontends

1. `run_policy.py`：拆 CLI/resolve、resource transaction、policy coordinator、execution session、debug observer。
2. `run_collect.py`：保留 headless canonical facade；下沉通用 episode/writer mechanics，保留 source-bound adapter。
3. 合并 deploy/collect 重复的 gripper/DMTAC profile resolution 到真实 owner。
4. `control_station/sim_hil.py` 形成 profile subpackage，application 仍是唯一 lifecycle coordinator。

### 阶段 5：重整公共包和测试树

1. 按稳定 domain family 为 ControlTrace、progress、version adapters、action execution、deploy RTC 建子包；
   保留受支持旧 import 的薄 shim。
2. 收窄 root `__init__`；先改变 repo 内新调用方式，再分批 deprecate，不一次性破坏数百个导出。
3. 将测试按 owner/contract/integration/hardware 层次组织；拆巨型测试但保留 marker、fixture 和命令入口。
4. 为 compatibility manifest 加静态测试，禁止新 shim 或反向依赖无登记增长。

### 阶段 6：资产、vendor 和 evidence 收口

1. vendor SDK 放到明确 vendor 边界，第一方 adapter 不与 vendor 实现混写；
2. 正式 asset 使用版本化 manifest/hash/获取方式；
3. test fixture、field evidence、operator calibration、generated report 和 local output 分目录；
4. 任何移动前检查 caller、设备/inode/硬链接和独立备份，尤其不批量改写 dataset/media。

## 7. 明确不推荐的做法

- 不新增 `assembly/`、`common/` 或新的通用 `utils/` 杂物层；模块必须对应真实 domain owner 或资源生命周期。
- 不按“每个文件不超过某个行数”机械切割；保持对象边界和不变量比行数重要。
- 不在结构重构中顺带统一 single/dual IK、改变 position/impedance、frame、solver hint、安全阈值或控制频率。
- 不为了快速让测试通过而扩大 `control_station` 反向依赖 allowlist。
- 不复制代码到新包后保留两份 active implementation；迁移必须只有一个 canonical implementation。
- 不让 `control_composition`、Station、sim profile 或 GUI 获得新的 authority、canonical safety、queue、cursor、
  dispatcher 或 episode commitment owner。
- 不把 `teleop_runtime.collection` 整体搬入 `collection_runtime`；source/device/UI-bound frontend 必须留在上层。
- 不一次性删除 549/337/266 个根导出；先取调用证据和兼容清单。
- 不清理 ignored assets、logs、teleop records 或现场 result，除非有明确 provenance、独立备份和删除授权。

## 8. 后续每批验收

最小静态/契约 gate：

```bash
/home/zc/anaconda3/envs/dexteleop310/bin/python -m pytest -q \
  tests/test_control_station_boundaries.py \
  tests/test_collection_runtime_boundaries.py \
  tests/test_control_composition_boundaries.py \
  tests/test_dataset_dependency_boundaries.py

/home/zc/anaconda3/envs/dexteleop310/bin/python -m py_compile <本批文件>
git diff --check
```

按风险追加：

- import-only，无设备/线程/socket 副作用；
- canonical entrypoint `--help`；
- fake backend 的 startup rollback、normal close、exception close；
- control order、Hold/Home、epoch/cancellation、single dispatch owner；
- pure validator 的 golden issue/order/hash 等价；
- Web package-data 的 wheel/install 后资源可发现性；
- targeted tests 后再跑与 owner 对应的宽回归。

真实硬件、motion、homing、replay、force-control 和 remote execution 不属于结构整理的默认验收，必须另行
获得明确范围和安全授权。

## 9. 本轮只读证据摘要

- 工作区状态已先检查，未清理或重置并行修改；
- AST 量化覆盖项目维护的 Python 文件、定义跨度、分支和跨包 import；
- 读取了 AGENTS.md、当前架构索引、各 owner README、real-teleop 历史重构临时文档；
- 深读了 dual-arm solver、real-teleop runtime/coordinator/loop、policy deploy、headless collect、position executor、
  RM65 backend/impedance、ControlTrace completeness、Sim-HIL 和 Web UI；
- 执行了纯静态/契约边界测试，未启动任何设备。

下一步建议先由用户确认本审计的优先级；若开始实施，推荐只批准“阶段 0：恢复边界与冻结契约”，不要
直接进入大规模文件移动。
