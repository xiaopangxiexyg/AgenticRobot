# 真机遥操与数采参数说明

运行命令见 [real README](./README.md)，完整参数语义见
[`teleop_runtime/README.md`](../../teleop_runtime/README.md) 和
[`teleop_runtime/collection/README.md`](../../teleop_runtime/collection/README.md)。

正式 teleop/collection 命令均为 M 级：可能连接并驱动机械臂、手或夹爪。单独的 `--help`、
`--validate-only` 和 VIVE sender 按各自入口标注的较低等级处理。

## 1. 启动顺序

1. 激活 `dexteleop310`，进入仓库根目录。
2. 停止占用相同 RM65、Q6、DMTAC、相机和 VIVE 端口的旧进程。
3. 执行设备 `--check`，确认左右 side、IP、串口、tool/TCP profile 和相机配置。
4. 按 [real README 的第 1.1 节](./README.md) 启动 VIVE sender。
5. 先做 virtual/calibration 验收；真机工作空间清空、急停可用后再启动 GUI。
6. GUI preflight 完成后人工放行，不使用未经审核的 auto-start 参数。

## 2. GUI human teleop / 正式采集入口

可直接复制的入口见 [real README 的第 1.2 节](./README.md)。它只是
canonical composition root；实际站点应显式提供设备、profile、dataset 和 debug 参数。
不要从旧日志中复制一串未命名的调参 flag。

## 3. 正式 position profile

当前正式 profile 是 `rm65_bimanual_position_v1`。它固定 15 Hz sample、60 Hz IK/qref、
120 Hz qcmd、accepted-qref FK main action、1.0/0.35 IK、adaptive min2/max4、position-only
奇异 branch 策略和低扰动数采配置。

`rm65_bimanual_position_v2_responsive` 是显式 opt-in 的响应性后继版本，不替换 V1。
它保持相同硬执行包络，仅增加健康普通区的按侧 post-IK fast path，并要求 q5 逃逸实际
进入 admitted qref 后才启用 release hysteresis。V2 必须使用独立 task/profile 名称；
切回 `POSITION_FORMAL_V1_ARGS` 即回退。

`POSITION_FORMAL_V3_QP_CONSTRAINED_ARGS` 是独立的 default-off 测试数组：它以 V1 为基线，
仅将既有正式 qref 速度/加速度区间下放到 Pink QP。测试数据必须使用独立 task/profile；
切回 `POSITION_FORMAL_V1_ARGS` 即完整回退。

正式可执行命令只在 [real README 的第 1.3 节](./README.md) 维护；本节负责
解释该命令选中的 profile 和验收字段。

参数数组只由 [`teleop_args.sh`](./teleop_args.sh) 加载，不会在 `source` 时启动设备。
旧 A/B、奇异问题定位过程和被替代命令保存在
[`docs/legacy/position_control/test-7.11.md`](../legacy/position_control/test-7.11.md)。

不要与 `POSITION_CONTROL_ARGS`、`POSITION_ACCELERATION_LIMITED_*`、`POSITION_IK_*`
叠加；这些是历史对照/调参数组，会覆盖正式 profile。

### 3.1 Formal profile 参数

| 层 | 当前值 | 应用范围 |
|---|---|---|
| Sample / IK / qref / qcmd | `15 / 60 / 60 / 120 Hz` | Live position teleop |
| Main action | 最后 accepted qref 的 FK pose | 写入 dataset；replay/deploy 不重复 pre-action 整形 |
| Inactive enter scale | `0.5` | 仅 live human source |
| Post-IK adaptive alpha | `0.14 → 0.45` | 仅 live human source、左右臂独立 |
| IK position/orientation | `1.0 / 0.35` | Pose action → qref |
| IK damping | `0.03` | Pose action → qref |
| IK iterations | adaptive `min=2 / max=4` | 60 Hz solve budget内 |
| Non-IK total reserve | `5.0 ms` | 其中 candidate certification `3.5 ms`，post-candidate `1.5 ms` |
| Candidate local clip | `8×` | Position-only candidate search |
| Singularity policy | wrist/elbow/shoulder 优先非奇异；无替代时允许认证后的奇异解 | 只影响位控 |
| qref velocity profile | `rm65_bimanual_position_v1_qref` | 不修改共享 RM65/力控 profile |
| Planner | `acceleration_limited` | Accepted qref → 120 Hz qcmd |
| qcmd max velocity | `5.6,5.6,7.2,12,12,16 rad/s` | 逐关节 |
| qcmd max acceleration | `56,56,72,120,120,160 rad/s²` | 逐关节 |
| Target-velocity filter | `tau=0.03 s` | 120 Hz planner |
| Velocity ramp | `0.10 s` | 120 Hz planner |
| Command step envelope | `0.675°` | 每次 qcmd proposal |
| Final follow alpha | `0.9` | 最终 120 Hz SDK qpos |
| Degraded speed | `0.5×` | 已分类异常目标，不改变正常 qdd 上限 |
| Reference max age | `0.2 s` | Stale 时 fail-safe hold |

参数真值位于版本化
[`robot_motion/rm65/position_profile.py`](../../robot_motion/rm65/position_profile.py)。
文档表用于运行检查，不应成为第二份配置实现。

### 3.2 Bash 数组职责与覆盖顺序

| 数组 | 责任 | 不负责 |
|---|---|---|
| `COMMON_REAL_TELEOP_ARGS` | 双臂、tool/TCP、碰撞与 GUI 基础配置 | 不选择 position/force profile |
| `DMTAC_CAPTURE_COMMON_ARGS` | DMTAC 数采公共选项 | 不决定左右并行策略 |
| `DMTAC_A_ARGS` | 当前推荐 DMTAC 调度方案 | 不改变 arm control |
| `DMTAC_TACTILE_WORKER_2CPU_ARGS` | 触觉 worker CPU `12,13` 与 nice | 不改变触觉数据语义 |
| `DATASET_COMMON_ARGS` | Dataset、相机、sample 公共配置 | 不改变位控参数 |
| `POSITION_FORMAL_V1_ARGS` | 选择正式 position profile 和 control mode | 不配置 writer/debug |
| `POSITION_FORMAL_V2_RESPONSIVE_ARGS` | 选择 opt-in 响应性 V2；硬执行包络与 V1 相同 | 不与 V1 task 混写 |
| `POSITION_FORMAL_V2_BALANCED_ARGS` | 选择冻结的 V2 平滑稳定基线；普通区 fast-path alpha=0.55 | 不与 V2 responsive task 混写 |
| `POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS` | 2026-08-11 真机验收后冻结的双臂纯位控发布面；profile SHA256=`dc97dc0e…0380` | 不开启平移/旋转 admittance；`POSITION_FORMAL_V2_BALANCED_ARGS` 是执行回退 |
| `POSITION_FORMAL_V2_BALANCED_60HZ_ADMITTANCE_ARGS` | 在 V2 Balanced 的 pre-IK target 上叠加已验收双臂平移+旋转 bounded admittance | 不修改 60 Hz IK、碰撞、qref、force guard 或数采 action 语义 |
| `POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ADMITTANCE_ARGS` | 在同一 Active 安全面上选择严格递增 source timestamp 的 120 Hz horizon owner | 不修改 60 Hz IK、碰撞、qref/qcmd 动力学或 force guard；JERK V1 数组是回退 |
| `POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_DAMPED_ADMITTANCE_ARGS` | 只对已有界的 60 Hz 平移/旋转 modifier 启用默认 `25 ms` output follower，抑制 20 Hz 受力保持的走停切换 | 不改受力映射、60 Hz IK、120 Hz qcmd 规划器或任何硬包络；去掉该数组即回退 |
| `POSITION_FORMAL_V2_SMOOTH_ARGS` | 选择默认关闭的残余微振动 A/B；普通区 fast-path alpha=0.45 | 不与 balanced task 混写 |
| `POSITION_FORMAL_V3_QP_CONSTRAINED_ARGS` | 选择 opt-in admitted-anchor QP 约束；其余使用稳定 V1 | 不与 V1/V2 task 混写 |
| `POSITION_FORCE_GUARD_CONSERVATIVE_ARGS` | 在正式 position 数组后恢复完整 `15 N / 2 Nm` 保守受力急停档 | 不改变 IK、qref 或执行 profile |
| `POSITION_COLLECTION_PROCESS_ARGS` | Spawn writer、共享内存槽、CPU `8-11` | 不改变 main action |
| `POSITION_COLLECTION_PRODUCTION_ARGS` | 关闭高频 JSON/FK debug，保留预分配数值 trace | 不关闭正式 dataset |
| `NO_COLLECTION_SENSOR_BASELINE_ARGS` | 禁用 collection/episode writer，形成控制与传感器基线 | 不禁用 RGB 相机或 DMTAC 触觉 |
| `CONSOLE_TIMING_ARGS` | 通过有界后台队列输出慢帧/stall，不启用 timing JSONL | 不提供正式性能结论 |
| `NO_DEBUG_DISK_WRITE_ARGS` | 关闭所有 standalone debug 写盘 | 不关闭正式 dataset |

Argparse 使用后出现的值覆盖前值。正式命令中不要在这些数组之后随意追加历史 position flag。

正式 60 Hz admittance 的可调入口是 source `teleop_args.sh` 前设置的六个环境变量：

- `POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT` 与
  `POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT`：相对真机验收基线的顺应倍率，默认
  `1.0 / 1.0`；Active 只接受 `[0.25, 4]`。它们不改变最终 `80 mm / 60 mm/s`
  平移 modifier 或 `5 deg / 9 deg/s` 旋转 modifier 包络。
- `POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC` 与
  `POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC`：当前 `profile_compliance` 真正生效的
  受力/扭矩输入阻尼，默认均为 `0.010857362 s`，值越大越平滑但响应越慢；Active
  上限为 `0.1 s`。
- `POSITION_ADMITTANCE_TRANSLATION_OUTPUT_FOLLOW_TAU_SEC` 与
  `POSITION_ADMITTANCE_ROTATION_OUTPUT_FOLLOW_TAU_SEC`：只被显式 `DAMPED` 数组使用，
  默认 `0.025 s`。这一阶跟随发生在已有界 modifier 上，不重复滤波原始受力；
  `0` 精确恢复现有行为，Active 上限为 `0.1 s`。无接触回程时输出只能逐分量
  及总范数向零收缩，不会追赶已撤力的旧 requested modifier。

正式 normal-parity Active 还要求 `420 mm/s + return_rate 4` 的无接触回程制动。
该速率只有在完整平移 modifier 逐分量且总范数单调回零时才会被最终边界采用；受力向外
运动仍受原 `60 mm/s` 上限约束。旧 `30 mm/s + return_rate 1` 日志仍可离线 replay，
但真机 Active 启动会拒绝该旧参数。

`--position-admittance-shadow-damping-n-s-m` 只属于旧 `second_order` 研究模型，在正式
`profile_compliance` 链路中不参与计算，不应把它当作已生效的正式阻尼参数。

### 3.3 启动后确认

Dataset `meta/info.json` 的 runtime metadata 至少应满足：

- `position_control_profile_resolved=rm65_bimanual_position_v1`；
- `main_action_contract=accepted_qref_fk_pose`；
- `position_ik_orientation_weight=0.35`；
- `position_ik_solve_min_iterations=2`、`position_ik_solve_max_iterations=4`；
- `position_ik_iteration_reserve_ms=5`；
- `position_ik_candidate_certification_reserve_ms=3.5`；
- `position_singularity_candidate_certificate=true`；
- `position_reference_hz=60`、`arm_control_hz=120`；
- `position_target_velocity_filter_tau_sec=0.03`；
- `position_force_guard.enabled=true`、`source=tool`；
- 默认 `position_force_guard.max_force_n=30`、`max_torque_nm=5`；追加保守数组时应为 `15/2`；
- `position_force_guard.stale_timeout_sec=0.4`；
- `position_force_guard.fail_closed=true`、`latched=true`、`stop_scope=bimanual`；
- `position_max_command_step_deg=0.675`；
- `collection_worker_mode=process`、`control_debug.level=off`；
- `actuator_trace=true`、`actuator_singularity_trace=true`。

启用正式 60 Hz admittance 时，还应确认
`position_admittance_response.schema_version=teleop.position_admittance_response/v1`、
`modifier_stage=pre_ik_target`、`ik_logic_modified=false`，并核对其中平移/旋转的
`response_weight`、`input_filter_tau_sec`、`effective_compliance_*` 和完整
`effective_config` 与本次采集记录一致。

`actuator_trace` v4 还会在同一个预分配 NPZ 中记录 60 Hz 的 pre-filter VIVE
`pose_vive_world`、OpenVR 原生线速度/角速度、各自 validity、tracker role 与采样/接收/
控制时间戳。速度字段只用于观测和离线分析，不进入遥操控制。

正常退出后还应确认 `collection_integrity_valid=true`、`submitted_samples == completed_samples`
以及 actuator trace 的 drop/error count 均为 `0`。

### 3.4 定向回退

只回退 shoulder/coaxial branch 纠正：

```bash
--no-position-prefer-non-shoulder-singular-branch
```

只回退 candidate 预算拆分和最终 qref 证书：

```bash
--position-ik-candidate-certification-reserve-ms 0 \
--no-position-singularity-candidate-certificate
```

只关闭稀疏 singularity trace：

```bash
--no-actuator-singularity-trace
```

这些是定向诊断回退，不代表旧 profile 重新成为正式配置。

### 3.5 Fix / Init 与数采边界

- 单臂精确归位到位后保持 Init 状态；操作员显式触发该侧 Home 重采后
  才恢复普通遥操。
- “双臂顺序精确归位”只执行左臂→右臂 qpos，不修改 tracker Home；
  双臂到位或任一步失败后进入常规 Hold。这个 Hold 需显式重采
  runtime Home 才能恢复，不会自动采 Home。
- GUI 可分别设置左/右臂精确归位速度倍率；`1.0` 对应
  `rm65_init_slow` 基线，默认均为 `5.0`。CLI 等价参数为
  `--init-arm-left-qpos-speed-scale` 和
  `--init-arm-right-qpos-speed-scale`。启动前归位、运行/数采过程归位、
  踏板归位和双臂顺序归位使用同一组倍率；归位进行中及 episode 忙碌
  期间不能修改。执行器按规划时间戳插值，不再按外层主循环调用次数推进，
  因而启动前、位控和力控链路不会因循环频率不同而改变归位速度。
- GUI、浏览器快捷键和终端 `t` 共用同一数采准入门：常规 Hold、
  Fix-Hold、exact-qpos 规划/执行/保持、Init 移动/Home 重采/失败恢复期间
  都不能新建 episode。已经打开的 episode 在 Hold/Fix-Hold 时暂停，恢复
  后进入新的 continuity segment。
- Fix/Init 的 Home 重采只计数新鲜且连续静止的 tracker 帧；网络中断、
  重复 timestamp、运动或采样窗漂移会继续等待或重置计数，不会将旧帧当成 Home。

详细标签、pause 和 continuity 语义见
[`DATASET_SCHEMA.md`](../../DATASET_SCHEMA.md#arm-mode-特殊状态标签)。

### 3.6 可选 Plus episode 整理

Plus 是外置、default-off 的 episode 级旁路，不改 RM65 raw writer、Parquet、媒体或 episode
成败。需要时显式启用：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui \
  --plus-episode-curation \
  --plus-annotation-root /path/to/plus_annotations
```

GUI 仅在开始前提供有限 `variant` 选择（配置 task profile 时）以及保存后一次
`disposition/reason` 提交；提交失败不影响已保存 episode。Plus 提交绑定当前
pending episode token，旁路写入期间不会允许删除/重编号。旧 v2.3 数据和显式的 v2.4
compatibility view 由同一 reader 读取，缺少 sidecar 时按“无 Plus 标注”处理。v2.4 native
capture identity 仍由上位 capture adapter 提供，Plus 不从目录名推断。

### 3.7 provisional v2.5-p 分块存储

v2.5-p native shadow 和它的 Parquet 分块物理后端均为 default-off，不改变现有正式命令。
需要离线/影子验证时，在原命令上同时增加：

```bash
--v25p-native-shadow \
--v25p-native-storage-layout parquet_stream_chunks_v1
```

RealTeleop thread/process writer 与 UMI 使用相同参数名。只启用
`--v25p-native-shadow` 时仍使用可回退的 `record_cjson_v1`；只指定 Parquet layout 而不启用
shadow 会直接报配置错误。该开关不声明 native-v2.5 eligibility，也不启用 v2.4 merge 或长程任务
GUI。启用后 action 经过共享 canonical `ComponentAction` 投影，v2.3 writer 仍是权威输出；
省略开关时不会构建 native store，也不会生成 `meta/v25p/native/`。

### 3.8 可选长程进度语义与独立 GUI

RealTeleop 的 coarse progress/Failure 语义和 Control Station 操作面均为 default-off，
不新增生产入口。现在有两种等价的选择方式。

当前 DMTAC RM65 正式任务可以在启动命令中固定初始 profile。在原正式命令末尾增加：

```bash
--collection-progress-profile catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress \
--collection-progress-gui
```

如果希望由数采员在 GUI 中从已有 managed profile 选择，只增加：

```bash
--collection-progress-gui
```

对新的空数据集，单独启用 GUI 时默认选择
`dmtac_rm65_bimanual_position_writing_four_stage_progress`；数采员仍可在首个 Episode
开始前切换其他 profile，显式 `--collection-progress-profile` 的优先级更高。已有 profile
绑定或已写入的 profile-less 数据集不受这个默认值影响，仍以数据集合同为准。

推荐真机使用第二种方式。完整操作顺序是：

1. 启动原数采命令并追加 `--collection-progress-gui`。右侧“数据集配置”模块中可修改
   保存根目录、正式 Task、英文名和中文名；点击“应用配置 / 切换新数据集”后，runtime
   会计算并切换到对应的最终 dataset 目录，但不会开始 Episode。
2. “长程任务进度”面板会在设备启动阶段出现；只要当前数据集尚未锁定且 catalog 中有
   managed LINEAR profile，选择器保持可选，不依赖正式 task/name 或 runtime Home。
3. 新目录没有 Episode/profile 绑定时，可以选择任意适用的长程 profile。选择本身不会
   开始 Episode；确认当前 Step 后再按原有数采开始键。也可以不选 profile 直接开始
   普通数采，长程进度不会阻塞首个 Episode。
4. 第一个带 profile 的 Episode 开始时 profile 才会绑定到 dataset。若先开始了无
   profile 的普通 Episode，该目录会保持 profile-less；后续要启用长程 profile 必须
   切换新的空 dataset。绑定完成后选择器变为只读，后续 Episode 和进程重启都继续
   使用该绑定。
5. 如果要换 profile，必须换用新的 dataset 目录，不能在原目录内混装。

数据集配置采用显式草稿/应用语义：在输入框中修改不会暗中切换 writer，也不会误开始
采集；只有点击应用按钮才提交。录制、暂停、倒计时或封口过程中配置会锁定，必须先结束
当前 Episode。保留相同 Task、只更换英文名时，`task_id` 不变，最终目录名会随英文名
变化，因此可以为同一正式任务建立一个新的独立数据集并重新选择该 Task 的 profile。

GUI 列出当前真机 adapter 已准入的全部 LINEAR managed profile，不按正式
`--collection-task-id` 或 `--collection-task-name` 过滤。profile 自己的 `task_id`、step
和 transition 仍是它的长程语义元数据；选择 profile 不会修改正式 task、数据集路径或
启动 Episode。显式文件路径 profile 仍可从 CLI 使用，但不会自动进入 managed GUI 列表。

当前 catalog 的 GUI 会列出包括以下 profile 在内的全部 LINEAR profile：

- `dmtac_rm65_bimanual_position_writing_four_stage_progress`：新空数据集的默认
  writing profile，依次为拿笔、取笔盖、书写、放回毛笔；
- `dmtac_rm65_bimanual_position_writing_five_stage_progress`：在四阶段流程中增加独立的
  沾墨阶段；
- `dmtac_rm65_bimanual_position_v1_progress`：单阶段正式 smoke profile；
- `test_dmtac_rm65_bimanual_position_v1_three_stage_progress`：显式标注为
  `【测试】` 的三阶段 GUI/Failure/Advance/Undo 测试 profile。

三阶段测试 profile 仍带有自己的 `task_id` 和测试语义；它可以出现在 GUI 选择器中，
但不是生产任务语义。必须使用全新/空白的最终 dataset 目录测试，不能把它选择到任何
已经由其他 profile 绑定的既有数据集。测试结束后，生产采集应回到真实任务自己的 profile
和独立 dataset 目录。

`catalog:example_three_stage_pick_place_progress` 仍是合同与后标注 refinement 演示样例，
它的 `task_id` 是 `example_three_stage_pick_place`，因此会作为独立选项出现在 GUI 中；
它仍不代表 DMTAC 的真实任务语义。需要真实三阶段任务时，应创建语义正确的 managed
profile，而不是依赖正式 dataset task/name 自动推断。

两个开关的责任如下：

- `--collection-progress-profile` 启用 progress Episode、Attempt、Advance/Undo/Failure
  事实和 Failure→Hold/Pause→Resume/Retry 接线，并提供初始选择；参数可以是
  `catalog:<profile_id>` 或显式 JSON 路径；
- `--collection-progress-gui` 显示 Control Station 进度插件并开放 profile 选择和
  typed progress event。未选择 profile 时仍允许普通 Episode 启动；如果该 Episode
  已写入，当前 dataset 会保持 profile-less，避免随后在同一目录混入有 profile 的
  Episode。

物理进度踏板的准入来自活动 Episode 的 progress 状态，不依赖进度面板是否显示。
因此通过 `--collection-progress-profile` 固定 profile 后，即使不启用
`--collection-progress-gui`，踏板 Advance/Failure 仍可工作；GUI flag 只控制面板与
浏览器事件。Episode 空闲、Failure/Hold、等待 Retry、末阶段或已有请求处理中，踏板会
fail-closed，并在日志中说明具体拒绝原因，且绝不会回落执行旧 Init 语义。

每个 coarse `progress_step` 还可以声明左右臂的 `arm_control`：`teleop` 表示该侧由
VIVE 遥操，`fixed_hold` 表示该侧机械臂和已配置末端一起保持固定。GUI 提供三种执行
模式，并且只允许在 Episode 空闲时切换：

- `profile`（默认）：进入新子任务时自动对齐声明的活动臂，但仍保留手动 Fix 和单臂
  踏板；
- `strict_profile`：严格对齐声明，禁用手动 Fix/单臂踏板，并在正式 writer 接纳样本前
  校验样本因果时刻的 `progress_step_id` 与左右 `teleop.arm_mode`/固定范围完全一致；
- `ignore_profile`：忽略子任务臂声明，只使用既有按钮和踏板语义。

Advance 若只减少活动臂会直接切换；若增加活动臂，则只在该侧已有新鲜、稳定的 VIVE
Home 候选时原子切换。候选不足时，系统保持旧子任务事实，先 Hold 并暂停正式数采；
暂停期间 writer 不接纳任何样本。下一次“开始遥操”只采目标新活动臂所需的 Home，建立
目标臂状态后才提交 Advance 并恢复写入，因此新子任务开头不会包含暂停样本。Undo 若
跨越不同臂声明，一律先进入相同的 Hold/Pause 边界，恢复时按回退后的 profile 建立活动
臂。Home 或精确 Fix 建立失败时保持 Hold/Pause 和原 progress step，不提交半完成切换。

progress 采集可选用额外的物理踏板快捷键，不改变 GUI typed intent 或
`collection_runtime` 状态机。四踏板仍兼容既有链路；当已绑定 progress profile 时，
踏板 3/4 分别复用 GUI 的“推进阶段”和“记录失败并 Hold”。五踏板的踏板 1/2
与双踏板完全一致：分别进入仅左臂遥操（Fix 右臂）和仅右臂遥操（Fix 左臂）；踏板 3
推进阶段，踏板 4 记录失败，踏板 5 撤销上一次进度推进。未选择 profile 时，踏板 1/2
仍可执行 Fix；踏板 3/4/5 保持 fail-closed，
不会回落到 Init，也不会排队等待后续 profile。某个 progress 操作在当前状态不可用时
也只会被拒绝，不会改触发其他语义。启动五踏板转发服务使用：

五踏板的 A 仍开始/停止数采；B/C 的派发与双踏板专用 B/C 完全一致：B 负责双臂遥操、
Hold、快速 Home 和单臂 Fix 释放，C 负责进入常规 Hold。网页和键盘都沿用同一套语义；
五踏板的 F15/F16/F17 只增加 progress Advance/Failure/Undo，不会覆盖 G/GUI 的 Init。

```bash
python -m hardware.foot_pad --count 5
```

安全测试会检查 1–5 全部踏板并在第五个键位确认后自动结束；测试本身不提交
Advance/Failure/Undo。第一次完整测试后会原子保存
`teleop_runtime/configs/hardware/foot_pad/five_pedal_test_receipt.json`，之后力控和位控
共用该验证结果，不再要求重复测试。该 receipt 同时绑定当前五踏板物理 assignment
指纹和遥操语义版本；重新标定、交换或更换任一踏板后会自动失效并要求重测。它只免除
GUI 安全测试，不替代 `hardware.foot_pad` 转发服务的启动和设备连接。

首个 progress-enabled Episode 启动前可以在 GUI 中更换选择。启动时，完整
`TaskProgressProfileV1` 会以 canonical 形式绑定到：

```text
meta/progress/profile.cjson
```

从这一刻起当前数据集只接受相同 semantic content hash；相同 `profile_id` 但内容或
版本不同也会被拒绝。重启后会从该文件恢复并锁定选择。已有 progress Episode 的旧
数据集只在所有 Episode-local profile 完全一致时允许安全回填；已有正式 Episode
但没有 progress profile 的数据集不能中途启用，必须使用新的数据集目录。因此一个
数据集不会混装不同 profile，也不会混装 profile-less 与 profiled Episode。

这里的“一个数据集”是一次正式任务对应的最终 dataset 目录，而不是可以同时容纳多个
日期/任务目录的上层 `--dataset-root` 容器。唯一绑定文件始终位于该最终目录下：

```text
<formal-dataset>/meta/progress/profile.cjson
```

常见拒绝及处理方式：

| 现象 | 含义 | 处理 |
|---|---|---|
| GUI 没有可选 profile | 当前 managed LINEAR catalog 为空或 adapter 未准入 | 检查 catalog/adapter；这不是由正式 task/name 匹配失败导致 |
| 未选择 profile | 当前 Episode 是普通 profile-less 数采，不启用长程进度 | 可继续采集；若要启用 progress，请在首个 Episode 前选择 profile，或切换新的空 dataset |
| 提示 dataset 已有 Episode 但未绑定 profile | 该目录已经是永久 profile-less 数据集 | 使用新的 dataset 目录 |
| 提示 profile/hash 冲突 | 目录已经绑定另一 profile，或同 ID 的内容/版本被修改 | 不覆盖旧文件；为新 profile 使用新目录 |
| 重启后选择器为只读 | 已从 dataset 绑定文件恢复 profile | 正常行为；继续采集即可 |

因此回退是分层且确定的：

- 只去掉 `--collection-progress-gui`、保留
  `--collection-progress-profile`：保留 progress 数据语义，但恢复原 RealTeleop
  页面和既有 A/B/C 行为；
- 同时去掉两个开关：不构建 progress workflow、不创建 `meta/progress/`，回到原正式
  数采链路；但不能用这种方式向已经绑定 progress profile 的同一数据集追加
  profile-less Episode。

当前真机 gate 只接受 `LINEAR` profile；Tree、profile 编辑器和 UMI 接入仍未开放。
已封存 Episode 的独立细粒度阶段标注 GUI 已开放，但它是离线 Control Station 应用，
不嵌入真机页面，使用方法见
[`LONG_HORIZON_STAGE_ANNOTATION.md`](./LONG_HORIZON_STAGE_ANNOTATION.md)。
统一 catalog 位于
`collection_runtime/progress/profiles/`，并明确分为 `linear/` 和 `tree/`：

- `catalog:dmtac_rm65_bimanual_position_v1_progress` 只有一个 terminal step，只用于
  单 Attempt/Failure/Retry smoke，因此没有可执行的“下一阶段”；
- `catalog:test_dmtac_rm65_bimanual_position_v1_three_stage_progress` 是当前 DMTAC
  task 可直接选择的测试专用三阶段 profile；所有 Step 均带 `【测试】` 标识；
- `catalog:example_three_stage_pick_place_progress` 是三阶段线性样例，依次为接近对准、
  稳定抓取、放置释放，并附带每个 coarse step 内的 refinement boundary 样例。

profile 的 `task_id` 只描述该 profile 自己的长程语义，不要求与正式
`--collection-task-id` 或 `--collection-task-name` 一致；正式 dataset 与 progress profile
仍分别保留各自的身份字段。样例不能冒充真实任务语义，真实长程任务应建立自己的唯一
profile/step ID，并在选择后由 dataset hash 绑定保护其内部一致性。
同名 `.stages.json` 定义封存后使用的有序执行阶段和 recovery 入口；它不会增加在线
Advance/Failure 按钮。旧 `.refinement.json` 保留为早期 salvage-boundary 兼容资产，
新细粒度 occurrence 标注使用 stage profile。

普通 Episode 从每个 Step 的首 stage 开始。补采从失败 stage 出发的 recovery 数据时，
在原真机命令末尾同时增加：

```bash
--collection-progress-profile catalog:<progress_profile_id> \
--collection-progress-entry-step <progress_step_id> \
--collection-progress-entry-stage <stage_id> \
--collection-progress-entry-condition recovery_from_failure
```

非首 stage 没有显式 `recovery_from_failure` 会在启动前失败。managed progress profile
若有同名 stage profile 会自动加载；显式外部 profile 可以另传
`--collection-stage-profile /absolute/path/task.stages.json`。首个 stage-profiled
Episode 会将同一 profile 锁定到 `meta/progress/stages.cjson`，同一数据集拒绝不同
stage profile 或 profiled/unprofiled Episode 混装。

不同任务可以复用 `APPROACH_TARGET`、`ACQUIRE_OBJECT`、`PLACE_OBJECT` 等
`semantic_type` 作为检索原语，但任务 profile 必须自己保存 step identity、显示名、顺序
和合法迁移，不能运行时继承一个可变的公共模板。

进度插件位于 `control_station/web/real_teleop_progress.js`，不拥有机器人控制、Hold、
Episode 或 writer。删除/禁用插件不会改变 progress 合同；`collection_runtime` 也不包含
任何 GUI 或真机设备逻辑。

### 空闲期触觉无损压缩（默认关闭）

正式 external-tactile 数采可在原命令末尾追加
`"${IDLE_TACTILE_COMPRESSION_ARGS[@]}"`（即 `--idle-tactile-compression`）。该功能不在
遥操/数采热路径压缩：只有设备启动完成，并且处于首次 Home 前或 Hold，同时不存在活动、
倒计时或 finalizing 的正式 Episode 时，才会以低优先级独立进程把已封存 raw tactile
流转换为逐块 delta+zstd 无损流。失败后暂停但仍保持 `is_recording` 的 Episode 不满足
准入条件。

开始 runtime Home 时先发出协作停止请求，已有 2 秒准备期用于完成当前小块；Home、快速
单臂切换或正式 Episode 真正开始前，还必须确认子进程已经退出并完成文件系统写入屏障。
切换 dataset recorder 和退出程序使用同一屏障。压缩按 manifest 原子提交，提交前会完整
解码并校验原始 SHA；只有 manifest 验证成功后才删除原始流。

完整回退无需迁移数据：不传该参数即可保持历史行为；也可在命令最后追加
`"${NO_IDLE_TACTILE_COMPRESSION_ARGS[@]}"`。已经无损转换的 Episode 继续由既有 reader
透明读取，不需要反向解压。该 overlay 要求 `--tactile-storage external` 和
`--tactile-compression none`，避免与同步热路径压缩混用。

### 采集期触觉 artifact hash（默认关闭）

正式 external-tactile 数采可以在原命令末尾增加：

```bash
--tactile-artifact-integrity
```

该参数让 collection writer 对实际落盘的 tactile stored bytes 增量计算 SHA-256，并在每个 Episode
完整封口后自动发布 detachable-modality inventory。hash 不扫描历史 payload，也不运行在机械臂控制
循环；raw、zstd、lz4 都按各自精确落盘表示绑定。回退参数为
`--no-tactile-artifact-integrity`，默认仍关闭。

该参数必须从一个新数据集的第一个 Episode 开始持续使用，或用于已有全部 stream 都已带
`stored_bytes_sha256_v1` 封存信息的数据集。对包含未封存历史 tactile 的旧数据集临时开启会在创建
任何新目录/新 Episode 前直接拒绝；请改用新数据集 root，或保持该参数关闭并在采集完成后走离线
全量扫描的 detach proposal。这样不会在每次 Episode 封口时反复扫描历史大文件。

该参数可以与 `--idle-tactile-compression` 同时开启。组合模式下，Episode 封口先发布 raw
tactile revision；空闲进程完成无损 zstd 转换后发布新的 current revision，下一 Episode 继续沿
同一 inventory 链追加。二者共同要求正式数采和 external tactile；空闲压缩还要求热路径
`--tactile-compression none`。若子进程不能在 2 秒内协作退出，系统会停留在首次 Home/Hold
边界并拒绝进入遥操，而不是在 artifact 状态不确定时继续采集；去掉任一参数即可独立回退。

启用该功能不等于允许在写入过程中拆目录。detach/rehydrate 仍是离线事务，并要求没有活动 writer、
reader 或 storage transaction。若目标数据集的 tactile 已合法拆出、手工缺失、部分缺失或 inventory
与当前字节不一致，RealTeleop 会在创建新 Episode 前拒绝启动正式 writer；必须先 rehydrate/修复，
不能让新 Episode 与冷端旧 Episode 混用同一个 `tactile/` 前缀。

采集期 inventory 只减少拆出前的准备和重复 hash 扫描，不替代 cold copy hash、独立 inode、restore
drill、destructive boundary 全量复核或 rehydrate 后完整 v2.3 validation。

“记录当前阶段失败并 Hold”是时间敏感事实：按钮在状态机准入后单击立即提交，不弹出
二次确认；前端 pending gate 会阻止重复点击。Failure 的事实持久化、Hold 和正式数采
暂停仍由原有 typed request/receipt 链路完成。

如果数采员因紧急情况先按了普通 Hold，且这次 Hold 确实在当前 progress Attempt 内
新建了正式采集暂停，则同一个未恢复的 Hold 状态下按钮会改为“将当前 Hold 记为阶段
失败”。单击后不会再次 Hold 或移动暂停边界：Failure 的 `effective_at` 沿用原
`formal_capture_pause.effective_at`，稍后的按钮时间记录为 fact `available_at`，
`timing_semantics=PROMOTED_HOLD_TIME`。原始普通 Hold 仍保留为真实应用事实，正式
pause segment 只追加 failure-promotion reason；progress receipt 把同一组已生效的
Hold/Pause 绑定到该 Failure。

这一提升只在“同一 Episode、同一 active Attempt、普通 Hold 后从未恢复、原暂停确由
本次 Hold 新建”时开放。已恢复、已换 Episode、数采在 Hold 前已因相机/模态健康暂停，
或由 tracker 丢失、碰撞/急停等自动安全路径触发的 Hold 都不会被回溯提升；Hold 中的
阶段推进和撤销也保持禁用。这样不会把更早的传感器暂停误当成失败边界，也不会把人工
调整时间写进失败 Attempt。

右侧操作区按“快捷状态、标定与运动参数、遥操流程、机械臂模式、数据集配置、
Episode 数采、长程任务、设备/输入/可视化”拆分为独立展示模块。可拖动模块标题改变
顺序，也可在标题获得焦点后使用
`↑/↓`；“恢复默认顺序”会清除自定义排列。顺序通过浏览器侧偏好保存，不产生
Control Station operator event，不改变 A/B/C、Hold、Episode、writer 或安全状态。

RealTeleop 与 UMI 共用的双栏页面现在默认允许调整栏宽。可拖动“当前状态”与“控制”两栏
之间的分隔条（方向键也可微调，`Home` 或双击恢复 60/40），窄栏内容会自动换行并折叠
内部网格；窗口小于 1100px 时仍沿用原有单栏布局。显式使用
`--no-gui-resizable-sidebars` 可完整回退到固定栏宽，且该功能不产生 operator event。

左侧“当前状态”顶部默认显示 A/B/C/G 与 P1–P5 小色块。浏览器快捷键按下时对应按键块
短暂变绿；运行时从虚拟脚踏键盘收到 press edge 时对应踏板块短暂变橙。它只消费已有
输入边沿并展示，不会产生新 operator event、改变事件优先级或写入数据集。使用
`--no-gui-input-activity-indicators` 可完整隐藏色块和其脚本；UMI 命令同样支持该开关。

## 4. Impedance/力控数采

力控没有被 position profile 替代。当前设备全开 15→40 Hz 命令只在
[real README 的第 1.4 节](./README.md) 维护；历史来源保存在
[`docs/legacy/position_control/test-7.11.md`](../legacy/position_control/test-7.11.md)。

Position-only profile、奇异 branch、qref 平滑和 position planner 参数不得加入此命令。

## 5. Headless/兼容 collection

入口命令见 [real README 的第 1.6 节](./README.md)。该入口保留给
headless/兼容场景。已有无相机、q6-only、arm-only 和单臂示例继续由
[`teleop_runtime/collection/README.md`](../../teleop_runtime/collection/README.md#5-最小启动命令)
维护。新正式 GUI 数采不要改走 `teleop_runtime.collection.run_collect`。

UMI 是 collection 的显式子命令，验证命令见
[real README 的第 1.7 节](./README.md)。`--validate-only` 仍会读取配置引用的标定文件；左右
`teleop_records/umi_hand_eye/latest_{side}.json`（`side=left/right`）缺失或过期时会明确
失败，应先完成两侧 `utils.umi_hand_eye`。验证通过后去掉 `--validate-only` 才正式启动；完整流程见
[UMI 数采](./UMI.md)。

## 6. Debug 档位

- 正式长程数采：`POSITION_COLLECTION_PRODUCTION_ARGS`，保留预分配 actuator/singularity
  数值 trace，关闭高频 JSON/FK 写盘。
- Session summary：仅用于检查整体时序。
- Full/session forensic debug：只用于短时复现，会与 60/120 Hz 控制线程竞争 GIL。
- `NO_DEBUG_DISK_WRITE_ARGS`：临时关闭 standalone debug，不关闭正式 dataset。
- 正式热路径和 30 Hz 实测状态 poller 只读取 RM65 轻量状态；完整 FK 与 timing
  percentile 快照留给低频诊断调用，不能重新放回 60/120 Hz 控制链路。

不要把 full debug 当成正式采集默认档。debug 档位数组和检查字段见
[`teleop_args.sh`](./teleop_args.sh)。历史 debug 组合见
[`docs/legacy/position_control/test-7.11.md`](../legacy/position_control/test-7.11.md)。
