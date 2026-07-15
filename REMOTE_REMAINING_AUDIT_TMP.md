# 远端剩余冗余与不对齐审计（临时）

状态：2026-07-15 对齐后更新。权威远端为
`user-B650M-AORUS-PRO-AX:/home/user/zc/DexTeleop`，本地项目根为
`/home/zc/HandMotion/Developing/DexTeleop`。同步方向是本地到远端的 no-delete 单向
同步：本地是代码源，远端是运行副本；远端独有文件默认保留，不按“镜像差异”直接删除。

## 本次复核的运行态与口径

- 本地 `realtime_sync_ssh.sh`/`lsyncd` 正在运行，源目录正是上述本地项目根，目标为
  `/home/user/zc/DexTeleop`，未使用 `--delete`。本次没有执行远端写入、删除、进程控制或
  真机控制。
- 远端复核时仍有 `python -m teleop_runtime.debug.run_vive_rm65_real_teleop` 真机遥操进程
  在运行。因此 `datasets/`、`teleop_records/`、`logs/`、硬件快照和根目录采集结果会变化，
  对这些目录的结论只能作为该时刻的快照。
- `remote_env.sh` 不激活环境时远端默认 `python` 不在 PATH；远端实际运行进程使用
  `~/miniconda3/envs/dexteleop310/bin/python`。这不是代码差异，后续远端测试必须显式指定
  `dexteleop310` 环境。
- 审计开始前本地工作树有 473 个状态项（194 个 `M`、102 个 `D`、177 个未跟踪项），
  所以本次以实际文件/哈希和同步结果为准，不能用 git commit 差异替代文件系统比较。
- `remote_rsync_project.sh --delete --dry-run` 的宽口径结果为：源侧 13,387 个文件、
  8,198 个待创建项、6,295 个待传输普通文件，远端若允许删除会报告 25,888 个删除项
  （18,845 个普通文件、7,043 个目录）。该命令未读取项目专用的输出/数据排除清单，数字
  会把数据集、运行产物和历史目录一并算入；它只是说明 no-delete 差异很大，绝不是删除计划。
- 目录版 `remote_sync_check.sh` 在本地和远端不同 locale 下会出现仅排序不同的假 mismatch。
  本次对 manifest 记录按路径/哈希/大小归一化后再判断实际差异；关键文件检查结果没有因此
  被误判。

## 当前结论

1. 核心代码没有发现新的同路径内容分叉。显式抽查的 23 个关键文件全部逐文件哈希一致；
   `dataset_contracts`、`dataset_interface`、`deployments`、`robot_motion`、`sim` 的可比
   源码、`utils/head_eye`、`utils/hand_eye`、`teleop_runtime/debug` 的共同文件也一致。
2. 仍有需要迁回/裁决的远端独有源码：3 个旧 VIVE receiver/visualizer 文件；它们已在
   本地归档，但权威远端原路径因 no-delete 同步仍保留，不能直接按残留删除。
3. 最大的实际差异来自远端历史硬件快照、远端实时采集结果、日志、测试分叉和被排除同步的
   `outputs/`。这些差异大多是证据或生成物，不是生产代码不对齐。

## 已完成并已对齐

- `assets/` 的四棵大资产树 `arctic_data`、`PND`、`robots`、`DM_Tac_SDK_Publish_1.2.10`
  已逐文件哈希复核一致。
- 远端 V1 兼容资产、2026-07-11 左臂基座 head-camera 标定、对应 head-eye provenance 和
  旧右臂基座封存包已同步到本地；当前已检查文件哈希一致。
- 两份 RealSense registry 已统一指向 `left_arm_base_link`、
  `teleop_records/headeye/20260711_155405_rm65_handeye.json` 及其 `T_camera_to_base`；
  远端同步检查通过。
- `cameras/` 与 `codex_remote/` 的已检查内容一致。主真机遥操入口
  `teleop_runtime/debug/run_vive_rm65_real_teleop.py` 当前与本地一致。
- OpenPI、部署与 RM65 安全姿态初始化原型簇已按本地基线定向清理；15 个旧原型文件及其
  2 个对应测试已删除，审查报告和快照见
  `logs/temp_branch/REMOTE_PROTOTYPE_CLUSTER_REVIEW_TMP.md`、`logs/temp_branch/remote_snapshot/`。
- `dataset_interface` 的远端训练相机 allow-list 雏形已按本地重构为逐 episode/视频视角有效性
  检查；`dataset_interface` 与 `tests/test_dataset_interface.py` 已回验对齐，本地定向测试
  8 passed。
- hand-eye / head-eye 分叉已在本地合并并与权威远端回验 `sync-check=ok`：固定相机 profile
  失败后的兼容选择、实际 profile 溯源、包级 API 导出和 wrist hand-eye 直接脚本导入均已
  收敛。详情见 `logs/temp_branch/HAND_EYE_HEAD_EYE_FORK_REVIEW_TMP.md`；标定 JSON/NPZ/NPY
  与 samples 未改写。
- 本次还确认 23 个关键文件逐文件哈希一致，包括运行入口、RM65/相机配置、关键 deploy/replay
  模块、hand-eye/head-eye 模块和 `utils.dataset_postprocess` 的共同 canonical 文件。
- `utils.dataset_postprocess` 的远端新增物已迁回并完成边界适配：2 个工具、3 个
  `geometry_mask` 兼容 shim、3 个 manifest 和 2 个配套测试已在本地加入。`organize_episodes.py`
  的 DmTac metadata 改为离线纯 JSON 描述，避免静态依赖 `robot_motion`；本地相关回归为
  `21 passed / 6 skipped`，远端实际 `dexteleop310` 解释器下为 `5 passed`。10 个新增源码/
  manifest/测试文件逐文件 `sync-check=ok`；目录归一化后该包无远端独有源码，剩余 144 个
  本地 `wrist_cam_adjust/outputs/` 文件是被排除同步的生成物。
- 旧 VIVE 工具已按当前框架拆分处理：`vive_tracker_benchmark_sender.py`、
  `vive_tracker_benchmark_receiver.py` 及配套测试已迁回并适配当前 sender 的“两枚
  GenericTracker、按设备名自动映射”契约；benchmark 改用独立 UDP `10002`，避开当前
  `10001` clock-sync 端口，定向 VIVE 回归为 `43 passed`。旧通用 receiver、固定 36-float
  receiver 和对应 visualizer 已保存在 `legacy/vive_udp_tools/`，不再作为 active import。

## 已完成审查并清理的远端源码

### `sim/contracts/scene.py` 拆包遗留

- 已将远端 15KB 单文件快照拉回
  `logs/temp_branch/sim_contracts_scene_remote_snapshot/scene.py`。
- 本地 `sim/contracts/scene/` package 已覆盖旧公共 API，并包含当前 MuJoCo 建模链路使用的
  `CollisionSpec.contact`；全仓本地调用均使用 package API。
- 已按“sim 以本地为准”定向同步 `/home/user/zc/DexTeleop/sim/contracts`，仅删除远端冗余
  `scene.py`，其余常规文件未传输、未改写；删除后 `sync-check=ok`。
- `tests/test_sim_contracts.py` 与 `tests/test_mujoco_collision.py` 共 8 passed。

### `dataset_postprocess` / `dexteleop.dataset` 历史 namespace

- 已归档远端旧代码至 `logs/temp_branch/dataset_postprocess_legacy_remote_snapshot/` 与
  `logs/temp_branch/dexteleop_dataset_legacy_remote_snapshot/`。
- 远端历史 Python 源文件已删除：`dataset_postprocess/{__init__,__main__}.py`、
  `dexteleop/{__init__,dataset/__init__,dataset/wrist_depth_codec.py}`。
- 本次再次查看权威远端：根目录 `dataset_postprocess/` 为空，`dexteleop/` 只剩
  `dataset/__pycache__/` 和根 `__pycache__/`，没有 `__init__.py` 或可执行旧实现，不能再提供
  上述历史 import/CLI 入口。
- 本地 `tests/test_package_migrations.py`、`tests/test_dataset_dependency_boundaries.py` 共
  12 passed。

### 其他已确认清理结果

- `portable_init_UV/` 在远端不存在。
- 旧 `sim/mujoco_real2sim/` 在远端不存在；当前远端与本地共同的是新的 `sim/mujoco/` 链路。
  因此这两组本地删除项没有重新出现在权威远端。

## P0：远端源码迁回/裁决记录

以下条目记录本次审计中需要优先迁回/裁决的远端源码；已完成项保留过程和验证证据，
未完成项在来源审查、迁移/封存和测试裁决前均不得删除。

### `utils.dataset_postprocess`：新增工具已完成对齐

- 当前共同 canonical 文件已哈希一致，包括 `cli.py`、`__init__.py`、`v21_to_v22.py`、
  `wrist_cam_adjust/depth_mask/postprocess.py`、`final_mask.py`、`__main__.py`、文档和脚本。
- 之前记录的“canonical 代码漂移”不再成立；远端新增物已迁回本地，并包含：
  `organize_episodes.py`、`relocate_mask_metadata.py`、`wrist_cam_adjust/{dataset.py,mask.py,video.py}`、
  3 个 manifest，以及 `test_dataset_organize.py`、`test_relocate_mask_metadata.py`。
- `organize_episodes.py` 从多个根目录按 manifest 选择 episode、重排连续 index、执行窄范围
  schema migration，并在发布前做 staging；`relocate_mask_metadata.py` 重写 mask JSON/JSONL
  中的绝对 artifact root，默认 dry-run 且不触碰二进制 bitpack；3 个 shim 转发到现有
  `geometry_mask` 实现。
- 为满足离线工具边界，`organize_episodes.py` 内的 DmTac slot metadata 使用本地纯 JSON
  描述，不静态导入 `robot_motion`，避免把数据后处理工具耦合到运动控制包。
- 迁回后本地相关回归 `21 passed / 6 skipped`，远端实际 `dexteleop310` 解释器下配套测试
  `5 passed`；10 个新增源码/manifest/测试文件逐文件 `sync-check=ok`。
- 本地另有 144 个 `utils/dataset_postprocess/wrist_cam_adjust/outputs/` 生成文件，远端当前
  没有同名输出目录；它们被 `.gitignore` 和 realtime sync 排除，不应与源码分叉混为一谈。
- 结论：目录归一化后该包无远端独有源码；保留上述本地输出和远端运行数据，不对它们执行
  no-delete 清理。

### hand-eye / head-eye 与掩码测试分叉（已完成合并并与权威远端回验）

- 已拉回远端快照至 `logs/temp_branch/handeye_headeye_remote_snapshot/` 并逐项审查。
- `utils/head_eye/camera.py` 保留 profile 查询与启动失败后的 fallback；正常 profile 成功时行为
  不变。新 samples/JSON 只增加可选的 `metadata.realsense_settings_report`，旧 JSON/NPZ/NPY
  继续兼容读取。
- `utils/head_eye/__init__.py` 已导出 profile 与 transform API；
  `utils/hand_eye/{run_calibrate_handeye,visualize_handeye}.py` 已采用标准 package-relative
  import，直接脚本和模块入口均经过 `--help` 回归。
- 权威远端仅使用 `/home/user/zc/DexTeleop`；此前的 `~/DexTeleop` 与字面目录
  `/home/user/~/DexTeleop` 都不是项目目录，仍禁止向它们同步或删除。

## P1：远端独有测试、兼容残留与待迁回分叉

### 旧 VIVE receiver/visualizer：已归档，远端原路径待清理

远端相对本地仍独有的 3 个源码文件为：

- `hardware/VIVE/mocap/udp_receiver_type.py`
- `hardware/VIVE/mocap/vive_data_receiver.py`
- `hardware/VIVE/mocap/vive_visualizer.py`

- `udp_receiver_type.py` 是无协议约束的通用 JSON UDP receiver；`vive_data_receiver.py`
  使用固定 36 个 little-endian float 的旧二进制协议，只保留最新数据；`vive_visualizer.py`
  依赖该二进制 receiver 和旧的 3 tracker/36-float 假设。三者均未发现当前仓内调用方，
  与 `teleop_runtime.vive/v1` JSON 主链路脱节。
- 三个原始文件已完整归档至 `legacy/vive_udp_tools/`，并附带迁移说明；当前本地 active
  package 不提供这些旧 import 路径。由于 realtime sync 是 no-delete，本次没有远端删除，
  权威远端原路径仍需在真机停止、确认无历史复现需求后用明确的 one-shot 清理处理。

### VIVE tracker benchmark：保留并按新框架对齐

- `hardware/VIVE/mocap/vive_tracker_benchmark_sender.py` 复用当前
  `vive_data_sender.py` 的 OpenVR 采样、tracker 映射和 pose 转换；
  `teleop_runtime/debug/vive_tracker_benchmark_receiver.py` 只处理独立 benchmark 协议，
  不连接机械臂、不发布控制指令。
- 该工具保留的有效指标包括分阶段 sender/receiver 频率、丢包/重复包、tracker 有效率、
  ACK RTT、sender 采样/读取耗时和 direct-link clock-sync 后的一程延迟，可用于判断当前
  新 sender/UDP 链路的容量上限。端口从旧的 `10001` 调整为 `10002`，避免和新框架的
  clock-sync 控制面冲突。
- 远端配套 `tests/test_vive_tracker_benchmark.py` 已迁回；本地 benchmark、diagnostic、
  sender、pose source 定向测试共 `43 passed`。

### 远端独有测试（本次确认共 7 个）

本轮 7 个文件已全部完成裁决，结论是“0 个按原文件迁回，2 个由本地 canonical
测试覆盖，1 个保留语义并按现行 API 改写合并，4 个因外部/缺失依赖暂不纳入”。

- **已由本地 canonical 测试覆盖（不复制远端文件）**：
  `test_mujoco_real2sim.py` 是本地 `tests/test_mujoco.py` 的子集，本地版本还包含
  当前 V3 URDF、材质和直接 qpos 等补充覆盖；本地 canonical 定向回归通过。
  `test_vive_motion_intent.py` 与本地 `tests/test_vive_intent_state.py` 逐字一致，
  因此保留本地命名和组织即可，不再生成第二份测试。
- **已合并有效语义（不保留原测试文件）**：
  `test_realsense_rgb_depth_recovery.py` 原样依赖已删除的
  `_set_missing_depth_frame_error`、missing-depth counters 和旧错误前缀；远端当前
  运行结果为 `2 failed`（`AttributeError`）。其中仍然合理的“连续 frame timeout
  进入 stale、达到阈值请求 restart，以及 stale 前只记录 runtime warning”已改写为
  当前 `_set_transient_or_stale_timeout_error` / `device_info()` 契约，并合并到
  `tests/test_realsense_rgb_startup.py`。本地相关回归为 `27 passed / 3 skipped`，
  远端 `dexteleop310` 回归为 `9 passed`。
- **暂不纳入当前本地测试集**：
  `test_inspire_arm_integration.py`、`test_inspire_data_collect_record_only.py`、
  `test_vis_window_modes.py` 均依赖仓外 `inspire_data_collect` namespace；远端的
  record-only 与 vis 测试实际 collection error 均为 `ModuleNotFoundError`。
  `test_openpi_local_deployment.py` 依赖本地和远端都不存在的
  `deployments.openpi.local`，不能凭测试文件反向补造实现。这 4 个应保留为来源审计
  记录，后续若重新引入对应外部包，再单独恢复到隔离的 integration/optional test 集。
- 本次不删除远端原测试文件：realtime sync 为 no-delete，且远端仍可能需要历史复现；
  以上 duplicate/stale/external 文件的远端原路径待后续明确 one-shot cleanup 或 archive
  policy 后再处理。

建议：测试迁移以本地 canonical 被测模块和可复现依赖为边界；只合并仍能表达当前行为的
断言，不为保留远端文件而回补已删除私有 API，也不为通过 collection 而引入仓外依赖。

## P1：远端生成的硬件状态，必须保留而非清理

`teleop_runtime/configs/hardware/{dmtac,q6,rm65_arm_ips}/` 当前相对本地多出 148 个时间戳
快照，逐文件归一化后分布为：`dmtac` 3 个、`q6` 1 个、`rm65_arm_ips` 144 个；时间范围
为 2026-07-08 至 2026-07-15。它们对应实际设备发现、IP 分配和手部配置历史。

- 当前共同的 `dmtac/latest.json`、`q6/latest.json`、`rm65_arm_ips/latest.json` 以及本地
  选定快照逐文件哈希一致；本次没有发现新的 `latest` 内容分叉。
- 真机启动可读取 `latest`，因此不能按 no-delete 残留删除；远端仍在运行真机程序，历史目录
  也可能继续增加。
- 后续应建立“当前 approved latest + 历史只读 archive”策略：先导出 manifest/来源，再选择
  迁回本地或迁到专用配置归档。

## P2：远端 legacy 文档与历史记录（根目录残余）

以下 7 个根级文件已从“待处理的普通残留”归类为远端 `legacy` 文档/记录资产。它们是
故障证据、临时参数、验收记录或设计草案，不属于当前 active package、生产配置或运行入口：

- `MUJOCO_LOG.TXT`：两次 GLFW 初始化失败的生成日志，保留作为环境问题证据。
- `temp_args_local.md`：人工加载 DMTAC 真机遥操作参数的临时 shell helper；远端内容声明
  不会执行命令，保留用于历史复现时核对参数来源。
- `temp.md`：RealSense Viewer 修复步骤记录。
- `temp_real_teleop_fault_investigation_20260707.md`：RealTeleop `FAULT` 卡住排查记录。
- `TEMP_V23_REAL_TEST.md`：v2.3 绝对/相对路径真机一致性临时验收记录。
- `test-7.12.md`：2026-07-12 修复与验收摘要，含“不执行真机运动”的记录。
- `VIVE_TRACKER_VELOCITY_POSITION_CONTROL_PLAN_TMP.md`：VIVE 速度接入与位置控制平滑改造
  的临时规划及实现状态记录。

建议的最终 archive 布局为：六个文档/临时 helper 放入
`legacy/root_docs/`，`MUJOCO_LOG.TXT` 放入 `legacy/root_logs/`，并在移动前保留
原始文件名、时间戳和一份来源 manifest。当前 realtime sync 为 no-delete，本次只完成
分类和审计，不移动或删除远端原路径；待确认真机与历史复现不再依赖后，再用明确的
one-shot archive/cleanup 处理。

## P2：有效运行产物，需归档/重定位而不是删除

### 数据与采集结果

`datasets/`、`teleop_records/`、`logs/` 不参与本次源码对齐判断；本地和远端各自保留运行
证据，目录不应被 no-delete 审计误当成生产代码。

- 本地 `datasets/` 当前有 16 个顶层条目，主要是 2026-07-01 至 2026-07-09 的旧真实采集、
  sample、`pi_template` 和 `test`；远端当前有 5 个不同顶层条目：
  `2026-07-11_dmtac_position_15_45_150hz`、两个 2026-07-12 position 数据目录、`masks`、
  `sorted_v22`。两侧顶层集合不重合是排除同步的预期结果，全部保护。
- 远端根目录另外保留 6 个本地没有的运行结果目录：
  `2026-07-11_dmtac_impedance_15_40hz`、
  `2026-07-12_dmtac_position_acceleration_limited_15_75_150hz`、
  `2026-07-13_dmtac_position_acceleration_limited_15_60_120hz`、
  `2026-07-14_dmtac_position_root_x_diagnostic_15_60_120hz`、
  `2026-07-15_dmtac_position_1x1_fullpath_8x_15_60_120hz`、
  `2026-07-15_dmtac_position_root_x_diagnostic_15_60_120hz`。其中 2026-07-11 和 2026-07-12
  的 `meta/episodes.jsonl` 非空；2026-07-13、2026-07-14 和两个 2026-07-15 目录复核时
  为 0 episode/0 frame 的采集骨架或诊断骨架。无论是否为空，都要先确认用途再归档。
- `teleop_records/` 顶层即时条目本地 67、远端 22，集合差异包括本地较新的 2026-07-11 至
  2026-07-15 会话以及远端的 `action_debug`、`control_debug`、`vive_diagnostics` 等目录；
  这些是校准、设备、debug、tactile 和运行证据，保留。

### 日志、缓存与输出

- 远端 `logs/` 在复核时约 3.5GiB，并有截至 2026-07-15 的 DmTac 日志；本地与远端日志顶层
  各约 69/68 个条目且集合不同。真机运行或未完成备份前不能删除。
- `sim/outputs/` 和本地 `utils/dataset_postprocess/wrist_cam_adjust/outputs/` 均为生成物：
  本地相对远端观察到 404 个 sim 输出文件，远端保留旧的 `mujoco_real2sim_20260703_195217`
  结果及一个 test XML（共 6 个文件记录）；远端还保留
  `sim/future_plans/real2sim2real_control_timing_alignment.md`。这些路径被输出/同步规则
  排除，不能当作源码分叉。
- `.cache/`、`__pycache__/`、`.pytest_cache/` 是可重建候选；只有在确认没有活动运行且不需
  要调试复现时才能清理。

## P2：当前明确不纳入本地生产基线的远端调试资产

按已确认范围，以下资产暂时保持远端，不同步、不删除：

- `assets/rm65_head_camera_old_new_comparison_tmp.urdf`
- `assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_visualization_mirror_{x,y,z}.urdf`
- `assets/rm65_inspire_gripper_v2/README.md`
- `assets/flange/robot.urdf`
- `assets/inspire_tac/meshes.zip`
- `assets/rm65_inspire_tac/rm65_arm_colored.urdf`
- `assets/rm65_STEP/rm_urdf_color_debug/` 下的 `fc_test_*`、`freecad_export_launcher.py`、
  `rm65_export_entry.py`

它们应在以后决定是否进入 `debug/` 或 `legacy/` 资产归档；当前不是生产对齐缺陷。

## P3：剩余空目录与环境残留

- `dataset_postprocess/` 和 `dexteleop/` 的空/缓存残留已在“已完成审查并清理的远端源码”
  中说明，不再与 legacy 文档混列。
- `.agents/`、`.codex/`、`.git/`：环境/仓库状态，不属于项目同步清理目标。
- 嵌套的 `utils/wrist_cam_mask_vis/` 远端确认为空目录，属于高置信清理候选，但仍应在
  确认没有活动工具依赖后处理。

## 推荐处理顺序

1. `utils.dataset_postprocess` 的 2 个工具、3 个兼容 shim、3 个 manifest 和 2 个配套测试已
   完成本地对齐；后续远端核验只使用 `/home/user/zc/DexTeleop`，严禁使用 `~/DexTeleop`
   或 `/home/user/~/DexTeleop`。
2. 7 个远端独有测试已完成裁决：2 个由本地 canonical 文件覆盖，1 个 recovery intent
   已按当前 RealSense API 合并改写，4 个外部/缺失依赖暂不纳入；后续只处理它们的
   来源归档和明确的 one-shot 清理策略，不再把远端原文件直接复制进本地。
3. 对硬件配置快照建立 approved/latest 与 archive 策略，禁止直接删远端历史。
4. 真机停止且完成备份后，将根目录采集结果、空数据骨架和 excluded outputs 迁到明确的
   archive/data 位置；不要用 `--delete` 解决它们。
5. 最后再处理缓存、空目录和已明确废弃的兼容 shim；本节 legacy 文档按上面的
   `legacy/root_docs/`、`legacy/root_logs/` 方案单独归档，并为保留的远端调试资产建立清单，
   避免下一次 no-delete 审计重复出现。

## 审计边界

本清单基于 2026-07-15 约 19:00–19:40（Asia/Shanghai）的远端顶层目录、只读 wrapper 检查、
显式 SHA-256 文件比较和归一化目录 manifest。`datasets/`、`teleop_records/`、`logs/`、
大部分 `outputs/` 被同步/版本规则排除，未对其全部嵌套内容做删除性判断；它们默认按有效
资产保护。远端真机程序仍在运行，任何动态数量（日志、会话、硬件快照、采集目录）都应在
归档前重新采样并记录时间戳。
