# 2026-07-22 运行命令清点（历史快照）

清点日期：2026-07-22。扫描范围为仓库内非 `legacy/`、非 `logs/`、非 `remote_data/` 的
Markdown，以及仓库拥有的 `__main__.py`/shell 入口。

本次共发现：

- 102 个 Markdown 文件；
- 91 个文件含命令式文本或代码片段；
- 67 种 `python -m ...` 写法，其中包含 `pytest`、`pip`、`mujoco.viewer` 等外部模块；
- 45 种直接 `python path/to/script.py` 写法。

这些数字不是“67 个 production CLI”。大量命令属于安装、测试、设备工具、离线分析、
历史验收或未来规划。

## 1. 当前入口等级

### Canonical

| 入口 | 用途 | 权威文档 |
|---|---|---|
| `teleop_runtime.debug.run_vive_rm65_real_teleop --gui` | GUI human teleop / 正式数采 | [真实系统](../../real/README.md) |
| `teleop_runtime.collection` | Headless/兼容 collection | [`teleop_runtime/collection/README.md`](../../../teleop_runtime/collection/README.md) |
| `teleop_runtime.replay` | 数据集浏览与 real replay | [Replay](../../real/REPLAY.md) |
| `teleop_runtime.deploy` | Policy deployment | [Deploy](../../real/DEPLOY.md) |
| `sim.mujoco.replay` | Canonical MuJoCo dataset replay | [仿真](../../sim/README.md) |
| `teleop_runtime.debug.run_vive_mujoco_teleop` | Real-like MuJoCo position teleop | [`sim/mujoco/README.md`](../../../sim/mujoco/README.md) |

### 受支持的分域入口

| 分域 | 入口族 | 说明 |
|---|---|---|
| OpenPI | `deployments.openpi.server`、`deployments.openpi.client.real` | 专用 server/client wrapper；artifact 为 I/O 权威 |
| UMI | `teleop_runtime.collection umi`、`utils.umi_hand_eye` | UMI collection 与 hand-eye 分开 |
| 设备 | `teleop_runtime.configure_devices` | 检查与权限/冲突修复副作用不同 |
| Q6/输入 | `hardware.teleop_hand.q6_assignment`、`hardware.angle_trigger`、`hardware.foot_pad` | assignment、标定和运行态输入 |
| RM65 init | `robot_motion.rm65.arm_init.qpos_init`、`qpos_sequence` | plan 生成与真实执行必须区分 |
| 数据 | `utils.dataset_postprocess` 及其子命令 | 默认 dry-run/是否写入以子命令为准 |
| Hand-eye | `utils.hand_eye`、`utils.head_eye`、`utils.umi_hand_eye` | 不同工装与 frame contract，不应混用 |
| Sim diagnostics | `sim.mujoco.run_control`、`sim.mujoco.diagnostics.*`、`sim.mujoco.benchmarks.*` | compile、诊断和 benchmark，不是新的 production runtime |

### 兼容或显式 debug 入口

| 入口 | 状态 |
|---|---|
| `teleop_runtime.collection.run_collect` | 兼容 direct module；新 shell 命令使用 `teleop_runtime.collection` |
| `teleop_runtime.deploy.run_policy` | 受支持的内部/direct 入口；shell canonical wrapper 是 `teleop_runtime.deploy` |
| `sim.mujoco.replay.runner` | Debug-only qpos/geometry replay；不等于 canonical action replay |
| `teleop_runtime.debug.run_vive_sapien_virtual` | SAPIEN preview；不是 task-physics sim |
| `teleop_runtime.debug.run_recorded_episode_replay` | Recorded episode debug，不替代 replay GUI |
| `teleop_runtime.debug.validate_recorded_episode` | 离线 debug validator |
| `teleop_runtime.debug.analyze_position_root_x` | Position X-axis 专项日志分析 |
| `teleop_runtime.debug.rm65_real_debug.*` | RM65 专项诊断；按其 README 判断是否只读 |
| `robot_motion.tests.*tracker_replay*` | 离线 replay 分析工具，虽然位于 tests package |
| `sim.mujoco_temp_pipeline.*` | 临时 train/evaluate/play pipeline |

### 当前不能作为入口

| 写法 | 原因 | 处理 |
|---|---|---|
| `python -m control_station` | 只在规划文档出现，当前没有 `control_station/__main__.py` | 保留规划，不写入运行配方 |
| `teleop_runtime/core/kernel.py` | AGENTS 明确标为 experimental | 不列为 production CLI |
| `python -m portable_dexhand021...` | 相关 README 假设从 `hardware/` 或包目录运行 | 保留原文；统一文档不宣称可从仓库根直接运行 |
| `python examples/...`、`python tools/...` | 多数依赖某个 vendor/package 子目录作为 cwd | 留在设备 README，不复制为根目录命令 |

## 2. 散落文档归属

| 原文档 | 当前职责 | 统一入口中的处理 |
|---|---|---|
| `README.md` | 站点完整流程与设备背景 | 保留；根部增加统一命令索引链接 |
| `README_ENV.md` | 环境唯一权威 | 保留安装细节；索引指向 runbook |
| `teleop_runtime/README.md` | runtime、calibration、dry-run 与真机语义 | 保留技术细节 |
| `teleop_runtime/collection/README.md` | headless/兼容 collection 完整参数 | 保留，不复制全部变体 |
| `teleop_runtime/replay/README.md` | replay GUI、安全与 real replay | 保留完整参数 |
| `teleop_runtime/deploy/README.md` | deploy contract、profile 和 A/B | 保留完整参数 |
| `sim/README.md`、`sim/mujoco/README.md` | sim ownership 与 MuJoCo 技术说明 | 统一 runbook 只选 canonical/debug 入口 |
| `hardware/*/README.md` | 设备专用操作 | 按 H1/H2/M 分类后链接，不搬运全部运动命令 |
| `utils/*/README.md` | 后处理、标定、可视化、统计 | 归入数据/硬件 runbook |
| `docs/tests/v2.3_test.md` | 版本验收 | 不视为日常启动命令 |
| `test-7.11.md` | 位控/力控专项演进与 A/B | 全文已迁入 `docs/legacy/position_control/`；根路径保留导航 |
| `temp_args.md` | Bash 参数数组 | 正式配方可 source；文件本身不执行命令 |
| `temp_dmtac.md` | 较早的 DMTAC 命令分层与 replay/deploy 实验 | 全文已迁入 `docs/legacy/dmtac/`；根路径保留导航 |
| `position_ab_report.md` | 历史离线结果 | 已迁入 `docs/legacy/reports/`；根路径保留导航 |
| `*_TMP.md`、`todos/` | 临时审计、架构计划和未来命令 | 保留，不进入当前入口表 |

## 3. 已核验内容

下列模块在当前 `dexteleop310` 环境执行 `--help` 通过，未连接硬件：

```text
teleop_runtime.debug.run_vive_rm65_real_teleop
teleop_runtime.collection
teleop_runtime.replay
teleop_runtime.deploy
teleop_runtime.debug.run_vive_rm65_calibration_debug
teleop_runtime.debug.run_vive_mujoco_teleop
sim.mujoco.replay
teleop_runtime.configure_devices
utils.dataset_postprocess
utils.umi_hand_eye
hardware.angle_trigger
hardware.foot_pad
robot_motion.rm65.arm_init.qpos_init
```

其他设备、GUI、vendor example 和运动命令只做了静态存在性/来源检查，没有批量执行。

## 4. 尚未迁移或待确认

- `README.md` 仍同时承担站点配置与长流程说明，后续可逐步缩成项目简介 + docs 导航；本次不删除。
- `docs/legacy/dmtac/temp_dmtac.md` 中的 75→150 Hz position 示例早于当前 60→120 Hz
  formal profile，仅用于历史对照。
- `docs/deployment/openpi.md` 和部分验收文档包含 `/home/user/zc/DexTeleop` 等机器路径；
  runbook 已改用 placeholder，原现场记录暂不改写。
- 从仓库根直接运行 `hardware/DMTAC_gripper/run_gripper.py` 时，当前源码布局需要
  `PYTHONPATH="$PWD:$PWD/hardware/DMTAC_gripper:..."`；进入 DMTAC 子目录并安装其依赖也是
  原 README 支持的方式。统一硬件 runbook 已写明前者。
- UMI `--validate-only` 不是“忽略现场文件的 parser smoke”：它会读取配置引用的左右最新
  hand-eye 标定。缺少 `teleop_records/umi_hand_eye/latest_<side>.json` 时失败是预期行为。
- Vendor/example README 中大量相对路径依赖进入子目录后执行；若未来仍频繁使用，应逐个改成
  “从仓库根运行”的完整路径，而不是一次性机械替换。
- 新生成的分析结果应统一进入 `logs/<case>/`；已有 position A/B 报告保存在
  `docs/legacy/reports/`，不再作为运行入口。

## 5. 后续治理建议

1. 给每个生产 CLI 增加稳定 `--print-effective-config` 或 `--validate-only`，让 runbook 不必
   通过启动设备来确认最终 profile。
2. 在 CI 增加 Markdown command lint：禁止把 `control_station` 等未实现入口标为 canonical，
   检查内部入口和生产入口的措辞。
3. 为真机命令统一 front matter 或固定小节：`等级 / 前置条件 / 命令 / 人工放行 / 输出 / 回退`。
4. 将站点参数固化在版本化 JSON/profile，Bash 数组只负责组合资源与 debug 档位。
5. 新实验命令先写验收文档；通过后再提升到 `docs/commands/<category>/RUN.md`，并在
   `REFERENCE.md` 说明替代关系。

## 附录 A：`python -m` 家族索引

下表覆盖本次 Markdown 扫描出现的仓库自有模块家族。`pip`、`pytest`、`py_compile`、
`serial.tools.list_ports` 和 `mujoco.viewer` 属于第三方或 Python 标准工具，不是 DexTeleop CLI。

| 家族 | 扫描到的模块 | 分类/去向 |
|---|---|---|
| Control station | `control_station` | 仅规划，当前不可运行 |
| OpenPI | `deployments.openpi.client.real`；另有 split-line 写法的 `deployments.openpi.server` | [OpenPI 部署](../../real/OPENPI.md) |
| Hardware input | `hardware.angle_trigger`、`hardware.foot_pad`、`hardware.teleop_hand.q6_assignment` | [硬件与标定](../../hardware/README.md) |
| VIVE | `hardware.VIVE.mocap.vive_data_sender`、`vive_diagnostic_receiver`、`vive_tracker_benchmark_sender` | VIVE README/CLOCK_SYNC；sender 为正常入口，其余为诊断 |
| RM65 trace | `hardware.rm65.utils.scripts.rm65_udp_actuator_trace` | Actuator identification 专项诊断 |
| RM65 init | `robot_motion.rm65.arm_init.qpos_init`、`qpos_sequence` | Plan 与真实执行分开；见 arm-init README |
| Robot-motion replay tools | `robot_motion.tests.compare_tracker_replay`、`view_tracker_replay`、`write_tracker_replay_report` | 离线分析，不连接真机 |
| Core teleop | `teleop_runtime.collection`、`teleop_runtime.replay`、`teleop_runtime.deploy` | Canonical/兼容入口表 |
| Compatibility direct | `teleop_runtime.collection.run_collect`、`teleop_runtime.deploy.run_policy`、`teleop_runtime.replay.web_gui` | 保留兼容；人工启动优先 wrapper |
| Replay arm init | `teleop_runtime.replay.arm_init.qpos_execute`、`qpos_sequence_execute` | M 级；仅按 replay arm-init README 执行 |
| Real teleop debug | `teleop_runtime.debug.run_vive_rm65_real_teleop`、`run_vive_rm65_calibration_debug`、`rm65_real_debug.run_rm65_real_state_viz` | 主入口、标定和专项只读/状态诊断需区分 |
| Virtual/debug teleop | `teleop_runtime.debug.run_vive_mujoco_teleop`、`run_vive_sapien_virtual`、`run_recorded_episode_replay`、`validate_recorded_episode`、`analyze_position_root_x`、`vive_tracker_benchmark_receiver` | 仿真 preview 或离线/网络诊断 |
| Device preflight | `teleop_runtime.configure_devices` | `--check` 与修复参数副作用不同 |
| Deploy acceptance | `teleop_runtime.deploy.acceptance` | 离线 A/B 验收 |
| MuJoCo build/control | `sim.mujoco.run_control` | MJCF/scene/qpos 诊断 |
| MuJoCo replay | `sim.mujoco.replay`、`sim.mujoco.replay.runner`、`sim.mujoco.replay.umi_runner` | Canonical、debug-only 与 UMI 专项分开 |
| MuJoCo diagnostics | `sim.mujoco.diagnostics.cube_drop`、`cube_lift`、`sim.mujoco.benchmarks.benchmark_cube_lift` | R1 诊断/benchmark |
| Temporary sim pipeline | `sim.mujoco_temp_pipeline.train`、`evaluate`、`play` | 实验 pipeline，不是 production CLI |
| Real2sim fit | `sim.rm65.real2sim.gripper_position_fit` | 离线系统辨识 |
| Dataset postprocess | `utils.dataset_postprocess`、`DM_force_calculate`、`real_undistort`、`wrist_cam_adjust`、`relocate_mask_metadata` | [数据与测试](../../data/README.md)；先看各子命令 dry-run 语义 |
| Hand-eye | `utils.hand_eye`、`utils.hand_eye.visualize_handeye`、`utils.head_eye`、`utils.head_eye.dual_arm`、`utils.head_eye.visualize_handeye`、`utils.umi_hand_eye` | 标定算法与 viewer；按工装选入口 |
| Background generation | `utils.random_background.download_openimages_sample`、`generate_assets`、`generate_samples`、`generate_wrist_sample_video`、`index_real_backgrounds` | 离线资产工具；可能访问网络或写大量文件 |
| Portable dexhand | `portable_dexhand021.tactile_viewer` | 文档使用包目录相对 import；根目录有效性待确认 |

## 附录 B：直接 Python 脚本家族

本次扫描到的 `python path/to/file.py` 主要分为：

- DMTAC：`check_usb_hub_distribution.py`、`run_gripper.py`、`run_gripper_only.py`、
  tactile base-frame calibration 和 actuator identification；以
  [`hardware/DMTAC_gripper/README.md`](../../../hardware/DMTAC_gripper/README.md) 为准。
- RM65：tool payload/TCP、force calibration、fixed-point impedance、teach record/replay、
  high-transfer debug；多数是 H2/M，统一索引不复制动作参数。
- VIVE：`hardware/VIVE/mocap/vive_data_sender.py` 是正常 sender；其他 benchmark/legacy
  路径按 CLOCK_SYNC 使用。
- Teleop hand/vendor examples：`hardware/teleop_hand/*.py`、`examples/*.py`、`tools/hardware_test/*.py`
  多数依赖进入对应硬件目录后运行，并可能驱动手部硬件。
- Portable visualization：`utils/portable_vis/*`，属于 R1 GUI/离线查看。
- Statistics：`utils/statistic/pose_action_lag_scan.py`、`pose_action_next_state.py`，属于离线分析。
- Wrist mask：`utils/wrist_cam_fixed_mask/painting/paint_video_mask.py`，属于离线 GUI 标注。

直接脚本没有统一参数规范。后续若某个脚本成为长期日常入口，应先补模块化 `python -m`
入口和稳定 `--help`，再提升到 runbook。
