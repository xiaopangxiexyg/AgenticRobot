# DexTeleop Dataset Schema

本文档描述 DexTeleop 当前使用的基础 dataset 结构。格式整体接近 LeRobot v2.1：每个 episode 的主时间序列存为 parquet，视频、触觉、腕部深度 mask 等大对象按 episode 外置存储，索引关系记录在 `meta/*.jsonl` 中。

## 目录结构

```text
datasets/<dataset_name>/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    devices/
      realsense.json
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    observation.images.<camera_key>/
      chunk-000/
        episode_000000.mp4
  tactile/
    observation.hand.<side>.tactile/
      chunk-000/
        episode_000000/
          manifest.json
          <sensor_key>.float32.bin
          <sensor_key>.uint8.bin.lz4
  wrist_depth/
    mask/
      observation.images.<camera_key>/
        chunk-000/
          episode_000000/
            manifest.json
            stream.bitpack.bin
```

`chunk-000` 按 `meta/info.json` 中的 `chunk_size` 分组。当前数据通常少于一个 chunk，但 reader 不应假设永远只有 `chunk-000`。

## 元信息

`meta/info.json` 是 dataset 级别总览：

- `format`: 当前为 `lerobot_v2.1_style`。
- `dataset_name`, `task_id`, `task_description`: dataset 和任务标识。
- `nominal_fps`: 采集目标频率，当前常见为 `15.0`。
- `episode_count`, `frame_count`: 当前 dataset 内 episode 和 frame 总数。
- `features`: 训练常用向量列的形状与 dtype，例如 `action`, `observation.state`, `observation.last_action`。
- `arm_action_specs_by_side`: 每侧手臂 action 语义，当前为 `xyz_quat`, `dof=7`。
- `hand_action_specs_by_side`: 每侧手/夹爪 action 语义。训练与部署不应硬编码维度，应读取 `active_dof`, `active_mask`, `slot_map`。
- `postprocess_history`: 后处理记录，可选。

`meta/tasks.jsonl` 记录任务 id 与描述。

`meta/episodes.jsonl` 每行对应一个 episode，核心字段：

- `episode_index`: 从 0 开始的 episode 编号。后处理抽取/删除后可保留原始编号，因此不保证连续。
- `episode_id`: 通常为 `episode_XXXXXX`。
- `frame_count`, `start_timestamp`, `end_timestamp`, `duration_sec`。
- `parquet_path`: 主 parquet 相对路径。
- `video_paths`: 各相机 mp4 相对路径。
- `tactile_paths`: 左/右触觉 manifest 相对路径。
- `wrist_depth_mask_paths`: 腕部相机 depth mask manifest 相对路径。
- `wrist_depth_raw_paths`: 腕部相机 raw depth manifest 相对路径。
- `metadata`, `summary`: 采集配置、频率统计、停止原因、故障状态等辅助信息。

启用 wrist depth mask/raw 采集时，`metadata.wrist_depth_integrity` 和
`summary.wrist_depth_integrity` 会记录 episode 级完整性标注。核心字段包括
`ok`, `failure_count`, `unseen_camera_keys`，以及每个 wrist camera 的
`missing_frame_count`, `missing_depth_image_count`, `missing_depth_scale_count`,
`mask_ready_count`, `raw_ready_count`, `failure_reason_counts`,
`failure_frame_indices_first`。读取端应将 `ok=false` 或任一 missing 计数大于
0 的 episode 视为 wrist depth 不完整样本。

遥操中按 hold 暂停正式数采时，当前 episode 会保持打开但暂停写样本；
resume 后继续写入同一个 episode。原始 `timestamp` 保持真实时间，不压缩
hold gap。`summary` 可包含以下暂停元信息：

- `recording_pause_schema`: 当前为 `dexteleop_recording_pause_v1`。
- `recording_pause_segments`: 每次 pause/resume 的起止时间、持续时间、原因、
  起止 `frame_index` 和 `step_index`。
- `recording_paused_duration_sec`: episode 内累计暂停时长。
- `recording_active_duration_sec`: 排除暂停后的有效写入时长，用于采集频率统计。
- `recording_wall_duration_sec`: episode 打开到停止的真实墙钟时长。

`meta/episodes_stats.jsonl` 是每个 episode 的统计摘要，常用字段包括 `frame_count`, `average_dt_sec`, `average_hz`, `mean_control_hz`, `fault_count`。
如果数据包含 pose/碰撞质量标注，还会包含 `pose_quality_*`,
`force_collision_detected`, `force_collision_label` 等字段。
历史数据可能没有这些质量标注字段；reader 应将其视为可选字段，而不是 schema 错误。

## Parquet 主表

每个 parquet 文件是一条 episode 的逐帧主表。基础列约定如下：

- `index`: 当前 dataset 展平后的全局 frame index，应从 `0` 开始连续递增到
  `frame_count - 1`。它是训练 row id，不是原始采集进程内 sample id。
- `episode_index`: 当前行所属 episode。
- `frame_index`: episode 内局部 frame index，从 0 开始。
- `timestamp`: 该训练样本的 action/sample 时间戳，单位秒。
- `session_state`: 采集状态，例如 `RUNNING`, `FAULT`, `HOLD`。
- `fault_codes`: 当前帧故障码列表。
- `action`: 训练用拼接 action 向量，维度由 `meta/info.json.features.action.shape` 给出。
- `observation.state`: 训练用拼接状态向量，维度由 `features` 给出。
- `observation.last_action`: 上一帧 action。

常见展开列：

- `action.<side>_arm.pose`: 单侧手臂目标 pose，通常 `[x, y, z, qx, qy, qz, qw]`。
- `action.<side>_hand.joint_target`: 单侧手/夹爪目标。有效维度以 `hand_action_specs_by_side` 为准。
- `action.replay.<side>_arm_joint_target`: 6 维机械臂关节目标，语义是采集时实际发送给
  RM65 backend 的 joint target。它是低层 `qpos_target` replay 的必需字段，也是默认真机 replay
  首帧 qpos init 的优先来源；若缺失，默认 replay 可退回
  `observation.arm.<side>.current_qpos` 或 `observation.arm.<side>.actual_hardware_qpos` 做首帧初始化。
  `Start` 阶段默认仍以 26 维 `action` 中的 7D arm pose 作为下游接口验证输入，不用该字段逐帧下发。
- `action.replay.<side>_hand_joint_target`: replay/可视化用的手部目标。手部实际维度仍以
  `hand_action_specs_by_side` 为准。
- 通用 replay 与后续模型部署应优先读取 26 维 `action` 和 action spec，
  再经同一套 `RobotAction`/`TeleopCommand` 解析路径下发。首帧 qpos init
  是机械臂定位辅助，不是模型 action schema 的一部分。
- `action.gripper.<side>.target`: DMTAC gripper 的标量夹爪目标，适合后续按 1 维 gripper 训练。
- DMTAC gripper 历史数据可能同时保留 6-slot hand 兼容向量；训练时应以
  `action.gripper.<side>.target` 作为每侧 1 维夹爪目标。其余 5 个 hand slot
  是无效 padding，可能为 `NaN`，不应仅因此判定整帧 action 损坏。
- `observation.arm.<side>.*`: 当前/实际关节、TCP、硬件回读和时序信息。
- `observation.hand.<side>.*`: 手/夹爪 telemetry、触觉索引、触觉采样状态。
- `observation.images.<camera_key>.*`: 视频帧索引、源时间戳、接收时间戳和视频路径。
- `teleop.channels.*`: 操作端输入通道时间戳和有效性。
- `teleop.pose_quality.*`: 从本帧已有 IK/防碰撞 metadata 派生的 pose 质量指标。
- `teleop.arm_mode.*`: 从本帧 command/session metadata 派生的左右臂语义状态标签。

读取时建议只依赖需要的列；调试/采集列可能随版本增加。
历史 parquet 可能没有 `teleop.pose_quality.*`，兼容读取时不应把缺失解释为坏样本。
历史 parquet 也可能没有 `teleop.arm_mode.*`；兼容读取时应按可选标签处理，
不要把缺失解释为 action/state schema 错误。

## 外置媒体

视频：

- 路径来自 `episodes.jsonl.video_paths`。
- parquet 中 `observation.images.<camera_key>.video_frame_index` 指向 mp4 帧号。
- 相机样本按采集时的 action timestamp 对齐；新数据应使用 `latest <= action_timestamp`，避免读取未来帧。

触觉：

- 路径来自 `episodes.jsonl.tactile_paths`。
- `manifest.json` 描述 `format`, `frame_count`, `sensor_keys`, 每个 sensor 的 binary stream 路径、dtype、shape。
- 新采集默认对外置触觉流使用 `lz4` chunk 压缩；也可以用 `--tactile-compression zstd|lz4|none` 覆盖。`none` 会写旧式裸 `.bin`。
- 压缩流在每个 stream spec 中记录 `encoding`, `compression`, `chunk_frame_count`, `chunks`。`chunks` 包含 `frame_start`, `frame_count`, `offset`, `compressed_bytes`, `raw_bytes`，用于随机读取单个 chunk。旧裸流可没有这些字段，reader 应按 `raw_c_order`/`none` 兼容。
- parquet 中 `observation.hand.<side>.tactile_frame_index` 指向 manifest 中的触觉帧。
- 触觉可能相对视觉/action 有延迟，reader 应使用 manifest 与 parquet 中的 timestamp/索引做一致性检查。

腕部深度 mask：

- 路径来自 `episodes.jsonl.wrist_depth_mask_paths`。
- `manifest.json` 描述 bitpack stream、图像 shape、每帧时间戳和有效深度比例。
- `stream.bitpack.bin` 使用 `np.packbits`，语义是指定深度范围内为 1，非法或范围外为 0。

## 训练读取约定

- `episode_index` 是 episode 身份编号，不是数组下标；删除/抽取后可能不连续。若后处理把一个 episode 切成多个 segment，可重新生成唯一的 `episode_index`，并在 metadata 中保留 `source_episode_index`。
- `index` 是当前 dataset 的扁平 row id；复制式后处理产物应重建 `index`，保证当前输出 dataset 内唯一且连续。需要追溯原始编号时，应使用派生 metadata 或额外 source 字段，而不是复用 `index`。
- `features` 和 action spec 是维度权威来源，不要硬编码 `action` 维度。
- 对 DMTAC gripper，新数据应按每侧 1 维 gripper target 训练；历史数据如果仍保留 6 slot hand target，需要通过 `action.gripper.<side>.target`、action spec 或后处理转换到训练所需格式。每侧未使用的 5 个 hand slot 是 padding，不应当作训练 action 维度。
- replay、部署和模型推理应共享同一个 action contract：arm 使用 7D pose，
  hand/gripper 使用 `hand_action_specs_by_side` 声明的 active slots。末端类型变化时，
  优先新增或更新 action spec 与后端适配器，不应在 replay、训练 reader 或 deploy
  入口分别硬编码维度。
- `FAULT/HOLD` 后继续写入的尾段、长时间卡顿分段、以及人工标记的碰撞 episode，应通过 `dataset_postprocess` 生成新的 clean/collision dataset，而不是原地改写原始数据。
- 后处理产物应更新 `meta/info.json`, `episodes.jsonl`, `episodes_stats.jsonl`，并保留 `postprocess_history`。

## Arm Mode 特殊状态标签

`arm_mode` 是逐帧、逐侧机械臂语义状态标注，用于让模型区分“正常遥操”、
“人为固定/保持不动”与“执行 init pose”。它是旁路标签，不参与 action
拼接，不改变 `action`, `observation.state`, `observation.last_action` 的形状或语义。

完整 payload 位于 `sample_metadata_json.arm_mode`，schema 名为
`dexteleop_arm_mode_v1`。parquet 中同时展开以下常用列：

- `teleop.arm_mode.schema`: 当前为 `dexteleop_arm_mode_v1`。
- `teleop.arm_mode.available`: `0/1`，表示本帧是否有 arm mode metadata。
- `teleop.arm_mode.<side>.id`: 训练友好的整数枚举。
- `teleop.arm_mode.<side>.label`: 字符串枚举。
- `teleop.arm_mode.<side>.is_special`: `0/1`，`fixed_hold` 和 `init_pose` 为特殊状态。
- `teleop.arm_mode.<side>.source`: 标签来源，例如 `normal`, `fix_arm`, `init_arm`,
  `recording_pause`, `session_state`, `inactive_side`。
- `teleop.arm_mode.<side>.tags`: 可扩展标签列表。当前可能包含 `fix_arm`, `init_arm`, `hold`。
- `teleop.arm_mode_json`: 完整 JSON，包含每侧 `mode/id/source/tags/detail`。

当前枚举：

- `unknown = 0`: 缺失或无法判断。
- `teleop = 1`: 正常遥操。
- `fixed_hold = 2`: fix-arm 固定侧保持上一帧下发目标动作。该标签不是硬件急停。
- `init_pose = 3`: init-arm 侧正在前往或保持 init pose。
- `stopped = 4`: hold、非 RUNNING session 或其他停止/暂停路径。

判定优先级：

1. 如果本帧处于 recording pause、hold 或非 `RUNNING` session，左右活动臂标为 `stopped`。
2. 否则该侧处于 init-arm active/recapturing，标为 `init_pose`。
3. 否则该侧处于 fix-arm fixed/recapturing，标为 `fixed_hold`。
4. 否则活动侧标为 `teleop`，非活动侧标为 `unknown`。

正式数采中触发 hold 时，当前 episode 保持打开但暂停写样本；同时 fix/init
特殊状态会停止。再次 hold/re-home 后恢复普通遥操，后续写入帧应从 `teleop`
重新开始，而不是继续继承 hold 前的 `fixed_hold` 或 `init_pose`。

兼容旧数据时，如果缺少 `teleop.arm_mode.*`，reader 应允许加载。需要补齐统一字段时，
推荐策略是：`session_state == RUNNING` 的活动侧按 `teleop` 填充；无法确认活动侧或
非 RUNNING 行按 `unknown` 或 `stopped` 填充，并在派生 metadata 中记录这是 legacy/default。

## Pose Quality 与碰撞标签

`pose_quality` 是逐帧 pose 质量标注；力/力矩超限碰撞只做 episode 级轨迹标签，用于后续筛选、统计或构造负样本。它们都是采集记录端派生信息，不参与 IK 求解、双臂防碰撞候选选择、轨迹限幅或真实下发逻辑。当前实现不会额外求 IK、不会额外跑 collision query，也不会新增力传感器读取；所有判断都来自该帧已经存在的 command metadata、faults 和硬件诊断文本。

兼容旧数据时，本节所有质量标注都应按可选字段处理。如果 `pose_quality` 或 `trajectory_quality` 不存在，默认不因为缺失而降权或判坏；需要补齐统一字段时，可按正常样本填充 `teleop.pose_quality.score=1.0`, `teleop.pose_quality.status=good`, `teleop.pose_quality.bad=0`, `teleop.pose_quality.available=0`，并在派生 metadata 中注明这是 legacy/default 兼容值。

### 逐帧 `pose_quality`

完整 payload 位于 `sample_metadata_json.pose_quality`，schema 名为
`dexteleop_pose_quality_v1`。parquet 中同时展开以下常用列：

- `teleop.pose_quality.score`: `0.0` 到 `1.0` 的启发式质量分。`1.0` 表示当前已记录信号下未发现风险；`0.0` 表示风险项较多或较重。
- `teleop.pose_quality.status`: `good`, `warning`, `bad`, `unknown` 之一。
- `teleop.pose_quality.bad`: `0/1`，只要存在 bad 级别原因即为 `1`。
- `teleop.pose_quality.available`: `0/1`，表示本帧是否有足够 metadata 进行判断。
- `teleop.pose_quality_json`: 完整 JSON，包含 `schema`, `source`, `reasons`, `bad_reasons`, `warning_reasons`, `metrics`。

输入信号按如下优先从 `TeleopCommand.metadata` 和每个 effector command 的 metadata 中读取：

- `collision_avoidance_status`
- `collision_distance_m`, `collision_min_distance_m`
- `collision_avoidance_fallback_reason`
- `ik_branch_candidate`
- `ik_jump_gate_status`
- `ik_time_budget_exceeded`
- `execution_step_metadata.step_limited`
- `ik_solve_pose_error.<side>.translation_m`
- `ik_solve_pose_error.<side>.rotation_deg`
- `joint_limit_margin.<side>.nearest_margin_deg`
- `joint_limit_margin.<side>.near_limit`

如果没有 command，记录为 `available=false`, `status=unknown`,
`score=null`, `reasons=["no_command"]`。如果有 command 但没有上述质量信号，记录为 `available=false`, `status=unknown`,
`score=null`, `reasons=["quality_metadata_unavailable"]`。

注意：上述 `unknown` 是新采集流程中“本帧存在但质量信号不足”的记录语义；旧 dataset 完全缺少 `pose_quality` 字段时，为了兼容训练和筛选流程，可以按正常样本默认处理，而不是把缺失字段解释成 `unknown` 或 `bad`。

有质量信号时，先生成 bad/warning 原因，再由原因决定状态：

- `bad`: `bad_reasons` 非空。
- `warning`: 无 bad 原因，但 `warning_reasons` 非空。
- `good`: 有质量信号，且没有 bad/warning 原因。

bad 级别规则：

- `collision_avoidance_status` 为 `hold_last_safe`, `unsafe_candidate`, `jump_gate_hold`。
- `collision_avoidance_fallback_reason` 非空。
- `collision_distance_m - collision_min_distance_m < 0`。
- 最近关节限位余量 `nearest_margin_deg < 3.0`。
- 最大 IK 平移误差 `max(translation_m) > 0.02` 米。
- 最大 IK 旋转误差 `max(rotation_deg) > 5.0` 度。

warning 级别规则：

- `collision_avoidance_status` 为 `boundary_active`。
- `ik_jump_gate_status` 存在且不是 `pass`。
- `0 <= collision_distance_m - collision_min_distance_m <= 0.02` 米。
- 最近关节限位余量 `nearest_margin_deg < 10.0`，或 `near_limit=true`。
- 最大 IK 平移误差 `max(translation_m) > 0.005` 米。
- 最大 IK 旋转误差 `max(rotation_deg) > 2.0` 度。
- `execution_step_metadata.step_limited=true`。
- `ik_time_budget_exceeded=true`。

分数计算公式：

```text
score = clamp(1.0 - 0.35 * len(bad_reasons) - 0.12 * len(warning_reasons), 0.0, 1.0)
```

这个分数不是物理可行概率，也不是控制器输入；它只是对“当前目标 pose 在已有 IK/碰撞保护信号下是否干净”的轻量记录端打分。若需要更严格的可达性 atlas、离线 dense collision checking 或任务相关打分，应作为后处理指标另行写入，不应替换此在线采集标注的语义。

### Episode 级力/力矩超限碰撞标签

力/力矩超限不做逐帧 `force_limit` 标注，也不写入 parquet 的逐帧训练列。原因是当前采集语义下，一旦真实下发过程中发生 hard-stop 力/力矩超限，轨迹会终止；因此只需要对整个 episode 做一次定性判断：

- `force_collision_detected=false`, `force_collision_label=clean`: 全程未检测到 hard-stop 力/力矩超限，轨迹在该维度上合格。
- `force_collision_detected=true`, `force_collision_label=detected_collision`: episode 内出现过力/力矩 hard-stop 超限，该轨迹可作为碰撞/超限负样本处理。

历史数据可能没有 `summary.trajectory_quality.force_collision_detected` 或 `force_collision_label`。兼容读取时，缺失即视为未记录到碰撞：`force_collision_detected=false`, `force_collision_label=clean`。这只是旧数据兼容默认值；若后处理能从已有日志或人工标注中确认碰撞，应以后处理结果为准。

episode 标签从已有诊断文本聚合得到，不新增逐帧力/力矩读数或阈值字段。当前识别来源包括：

- parquet 的 `faults_json`。
- `observation.arm.<side>.actual_hardware_result_text`。
- `observation.arm.<side>.actual_state_error`。

匹配的典型文本形态为：

```text
force <value>N > hard_stop_force_n
torque <value>Nm > hard_stop_torque_nm
```

例如 `RuntimeError: left: force 55.0000N > hard_stop_force_n` 会把该 episode 标为 `detected_collision`。这里的 `detected_collision` 是基于力/力矩 hard-stop 的定性标签，不等同于几何碰撞检测；几何层面的风险仍应看 `pose_quality.metrics.collision_*` 与相关 bad/warning reasons。

### Episode 级汇总

`episodes.jsonl` 的 `summary.trajectory_quality` 汇总逐帧质量标注：

- `pose_quality_status_counts`: 各 `status` 的帧数。
- `pose_quality_bad_count`, `pose_quality_warning_count`: bad/warning 帧数。
- `pose_quality_min_score`, `pose_quality_mean_score`: 本 episode 的 pose quality 最小/平均分。
- `force_collision_detected`: episode 内是否出现过力/力矩 hard-stop 超限。
- `force_collision_label`: 若出现过超限则为 `detected_collision`，否则为 `clean`。

为方便快速筛选，`episodes.jsonl.summary` 顶层也会写入
`force_collision_detected` 和 `force_collision_label`。`episodes_stats.jsonl`
同步写入同名统计字段。
