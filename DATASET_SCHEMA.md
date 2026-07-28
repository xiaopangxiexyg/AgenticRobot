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
    modality_validity.json
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
  streams/
    <stream_id>/
      chunk-000/
        episode_000000/
          samples.cjsonl
          payloads/<payload_hash>.cjson
  meta/v24/native_stream/
    contracts/<stream_id>.cjson
    episodes/<episode_uuid>/manifest.cjson
  meta/v25p/native/
    dataset_manifest.cjson
    profiles/{embodiment,component_stream,task,compatibility_projection}/<hash>.cjson
    episodes/<episode_uuid>/
      identity.cjson
      embodiment_binding.cjson
      streams/<stream_id>/<sequence>.cjson
      samples/<sample_index>.cjson
      native_capture_manifest.cjson
      episode_manifest.cjson
```

`chunk-000` 按 `meta/info.json` 中的 `chunk_size` 分组。当前数据通常少于一个 chunk，但 reader 不应假设永远只有 `chunk-000`。

`streams/` 是可选的 v2.4 native-stream capability，不替代现有主 parquet、video、tactile 或 wrist-depth
布局。每个 stream 必须先绑定不可变 `NativeStreamContract`，明确 producer、value schema、shape/dtype、units/frame、
measured/available clock、rate/retention/validity 和 access role。sample index只保存stable sample ID、sequence、时间、
valid/stale/gap、payload object ID与hash；payload按content hash去重。episode manifest封存所有stream index hash、
sample/drop/gap统计，封存后不可追加。inline stream已能与同UUID/index的ControlTrace seal及capture commit绑定。
由既有video/tactile writer最终落盘的external payload另使用两阶段契约：capture-time intent先冻结sample identity，
legacy finalize完成后再提交dataset-relative artifact locator、logical payload hash、artifact hash/size及durability receipt；
receipt完整并经本地复核后native manifest才seal。reader对external sample返回verified locator，不把容器或压缩块伪装成
已解码array。当前两阶段路径只有fake video/tactile artifact验证，production specialized writer尚未生成receipt，
shared-memory sample identity和GUI/真机profile也未接通；因此仅存在stream文件仍不能单独证明完整v2.4 capture。

`meta/v25p/native/` 是默认关闭的 v2.5-p 原生准备层，不以 Parquet 根 `action`/
`observation.state` 或 v2.3 episode row 作为权威。`EmbodimentCaptureProfile` 描述本体组件，
`V25pComponentStreamProfile` 再为每个 action/state/observation/audit stream 绑定 component、
语义阶段、payload schema、表示、dtype/shape/frame 与存储方式。每个 causal sample 只引用在 cut
时可用的不可变 component record。小向量使用 `inline_tensor`；媒体等大载荷使用带内容哈希、
encoding 和 selector 的 `external_artifact` 描述符。

旧 `meta/v25p/episodes/` 与 `dataset_contracts.v25p` 属于冻结的 v2.3-based compatibility v1，
不是上述原生格式。v2.3 输出只能由显式 adapter 从原生 component stream 投影；adapter 必须按
component/role/stage/schema 唯一匹配并报告被省略的原生 stream。当前原生层仍标记 provisional，
也不宣称最终 native v2.5。RealTeleop thread/process writer 与 UMI recorder 已接入同一条默认关闭
的 producer 边界：源 sample 先构造 canonical `ComponentAction`，再由 `dataset_interface` 的显式
投影器生成 V2 action component streams；state/qpos 仍由各 source adapter 按原始时序附加。只有
显式传入 `--v25p-native-shadow` 才会创建这些文件，默认正式命令继续只写既有 v2.3。dual-write
gate 会把原生记录再次显式投影为 v2.3 component fields，并与旧 encoder 独立生成的 row 按声明
dtype/shape 比较；complete receipt 也只能在既有 episode owner 完成 legacy finalize 后发布。
后续仍需长时间真机 shadow 证据和 v2.4 commit 合并 gate。

训练用的 wrist final keep-mask 通常是离线后处理派生产物，不要求写回原始
dataset 目录。默认位置为：

```text
utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask/
  <dataset_name>/
    episode_000000/
      <camera_key>/
        manifest.json
        final_keep.bitpack.bin
```

如果需要把 final mask 路径固化进 dataset metadata，可在 `episodes.jsonl` 中添加
`final_mask_paths`、`wrist_final_mask_paths` 或 `wrist_keep_mask_paths`，结构与
`video_paths` 相同：`camera_key -> manifest path`。读取端应优先使用显式路径；
没有显式路径时，可按上面的默认 final mask root 解析。

## 元信息

`meta/info.json` 是 dataset 级别总览：

- `format`: 当前为 `lerobot_v2.1_style`。
- `dataset_name`, `task_id`, `task_description`: dataset 和任务标识。
- `nominal_fps`: 采集目标频率，当前常见为 `15.0`。
- `episode_count`, `frame_count`: 当前 dataset 内 episode 和 frame 总数。
- `features`: 训练常用向量列的形状与 dtype，例如 `action`, `observation.state`, `observation.last_action`。
- `dexteleop_dataset_schema`: 若为 `dexteleop_dataset_v2.2`，表示下面的 v2.2 policy-I/O
  contract 生效；目录布局仍是 `lerobot_v2.1_style`，不是上游 LeRobot 格式版本升级。
- `policy_io_layout`, `policy_io_layout_hash`: v2.2 的唯一维度/语义权威和其 canonical SHA-256
  标识。dataset、episode metadata、训练 checkpoint 与部署模型服务必须使用同一个 hash。
- `arm_action_specs_by_side`: 每侧手臂 action 语义，当前为 `xyz_quat`, `dof=7`。
- `hand_action_specs_by_side`: 每侧手/夹爪 action 语义。训练与部署不应硬编码维度，应读取 `active_dof`, `active_mask`, `slot_map`。
- `postprocess_history`: 后处理记录，可选。

`meta/tasks.jsonl` 记录任务 id 与描述。

`meta/modality_validity.json.capture` 是采集阶段的数据集级模态契约。它声明整个数据集明确
忽视的 camera key 和 tactile side；该规则不写成 episode 属性，也不允许同一数据集中的
episode 使用不同规则。只有全新/空白数据集可以首次选择或改写规则；已有数据的目标数据集
只允许使用与现有声明完全一致的规则。没有声明的历史非空数据集按“没有忽视项”解释。训练端
应先读取该声明选择数据集，再决定模型输入模态；详见 `dataset_interface/README.md`。

`meta/episodes.jsonl` 每行对应一个 episode，核心字段：

- `episode_index`: 从 0 开始的 episode 编号。后处理抽取/删除后可保留原始编号，因此不保证连续。
- `episode_id`: 通常为 `episode_XXXXXX`。
- `frame_count`, `start_timestamp`, `end_timestamp`, `duration_sec`。
- `parquet_path`: 主 parquet 相对路径。
- `video_paths`: 各相机 mp4 相对路径。
- `tactile_paths`: 左/右触觉 manifest 相对路径。
- `wrist_depth_mask_paths`: 腕部相机 depth mask manifest 相对路径。
- `wrist_depth_raw_paths`: 腕部相机 raw depth manifest 相对路径。
- `final_mask_paths` / `wrist_final_mask_paths` / `wrist_keep_mask_paths`: 可选，
  腕部相机 final keep-mask manifest 路径。没有这些字段时，训练接口可从外部
  final mask root 按 `<dataset>/<episode>/<camera>/manifest.json` 解析。
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
- `recording_pause_reasons`: 当前未结束 pause 窗口内按因果顺序去重的
  全部原因；已结束窗口的同等信息位于
  `recording_pause_segments[*].reasons`。`reason` 仍保留第一个触发原因以兼容旧 reader。
- `recording_paused_duration_sec`: episode 内累计暂停时长。
- `collection_continuity_segment_count`: episode 内不跨暂停边界的连续片段数。
- `recording_active_duration_sec`: 排除暂停后的有效写入时长，用于采集频率统计。
- `recording_wall_duration_sec`: episode 打开到停止的真实墙钟时长。

每条 Parquet row 的 `collection.continuity_segment_index` 从 `0` 开始，每次
pause/resume 完成后加 `1`。resume 同时清空 `last_action` 和 init audit 时序
缓存；`dataset_interface.pose_policy_window_from_rows` 必须拒绝跨 segment
组窗，避免模型把暂停前后的两段动作误认为连续轨迹。旧数据缺失该列时
按 segment `0` 兼容。

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

## DexTeleop v2.2 policy-I/O contract

v2.2 保持 LeRobot v2.1 风格的目录、episode metadata、parquet 和外置媒体；只替换模型
低维输入/输出的定义。启用条件是 `meta/info.json.dexteleop_dataset_schema =
"dexteleop_dataset_v2.2"`，并且 `policy_io_layout_hash` 与嵌入的 `policy_io_layout` 自洽。

- `action` 是 float32，固定前缀始终为 `left_pose7 + right_pose7`（14 维，`xyz + quat_xyzw`），
  再按 `left`、`right` 的 layout 顺序拼接**连续型**末端 target。没有配置末端的一侧不占维度。
- `observation.state` 同样以两个 arm pose 的 14 维前缀开始，再按 layout 拼接连续型末端
  measured state。默认 state channel 复制 target channel，因此默认 state/action float 维度相同；
  需要不同 state 维度时必须在 layout 明确声明。
- 离散末端绝不写进上述 float 向量。`action.discrete`、
  `observation.state.discrete`、`observation.last_action.discrete` 是独立 fixed-size
  `int64` token 列；每个 token 的合法 category、label 与可选 backend value 在 layout 中声明。
  reader/training batch/deployment model input 必须保留 `int64`，禁止转换成 float feature。
- `observation.arm.qpos` 固定为 `left_qpos6 + right_qpos6`（12 维 float32）且标记为
  `audit_only`；`observation.arm.qpos_valid` 是对应 bool mask。它不属于默认
  `observation.state`，dataset interface、训练 batch 与部署 observation 默认不读取它。
- 组件审计列包括 `action.arm.pose`, `observation.arm.pose`，以及每侧
  `action.end_effector.<side>.target` / `observation.end_effector.<side>.state` 或相应
  `_token` 列。它们便于审计和 adapter 调试，不替代根 `action`/`observation.state`。

每侧末端由 `policy_io_layout.end_effectors.<side>` 独立定义，可为 `none`、`continuous`
或 `discrete`。例如左侧 `none`、右侧 6D Inspire hand，或两侧不同类型的 1D gripper/离散
gripper 都是合法 layout。`source_indices` 只表示采集时从历史/raw 向量取值；如果需要复用
当前 q6 deploy/replay 路径，必须另行声明 `legacy_q6_slot_indices`。代码不会从采集 source
索引推断硬件命令映射。

一个可直接使用的左右不对称样例位于
`teleop_runtime/configs/policy_io_layouts/dual_arm_left_none_right_inspire_v22.json`；当前双
DM-Tac 夹爪使用
`teleop_runtime/configs/policy_io_layouts/dual_arm_bimanual_dmtac_gripper_v22.json`。真机采集
推荐用 `--policy-io-layout auto`：它只根据已实例化的 backend/retargeter 成对解析 canonical
layout，并选择语义完全相同的 preset；未知、缺 backend、缺 retargeter 或显式 JSON 不一致时
fail-closed。`--policy-io-layout <json>` 是显式复现覆盖，同样必须与 runtime 合同一致。未提供
该参数时保持旧 v2.1 行为。v2.2 采集会在 `meta/info.json` 与 episode metadata 写入
`policy_io_layout_collection`，包括 `auto_preset`/`auto_generated`/`explicit` 来源和 strict 标记。

严格 v2.2 采集不允许把“上次下发命令”伪装为 state：DM-Tac 必须记录新鲜
`gripper_measured_scalar`（带 `joint_timestamp`），Inspire 必须记录新鲜
`measured_joint_pos`。缺失、非有限或超过 layout 声明 freshness 上限的值会拒绝该 row；不会
回退到 legacy q6 state。DM-Tac 的 action/state 统一为归一化 `0..1`，历史 profile 中的
`0.9`/`0.95` 上限会在 runtime 中归一为 `1.0`。

旧 fixed-q6
数据可先用如下 copy-on-write 命令生成新目录，绝不原地改写源 dataset：

另一个 `dual_arm_left_gripper_right_discrete_v22.json` 展示左侧连续 1D gripper、右侧
离散 `int64` token 的组合；离散 category 的 `backend_value` 仅供显式 legacy-q6 bridge
使用，模型本身看到的仍是 token id。

```bash
python -m utils.dataset_postprocess v21-to-v22 \
  <source_dataset> --output-root <new_v22_dataset> --layout-json <layout.json> --apply
```

转换 layout 必须明确选择 legacy q6 中哪些 slot 代表真实末端，特别是 1D gripper；转换器不会
猜测 padding 的语义。

部署时，v2.2 policy/server metadata 必须同时声明 `policy_io_layout_hash`、连续 action
dim 和离散 action token dim。OpenPI adapter 会把 state token 作为独立 `state_discrete`
`int64` 请求字段发送，并要求响应的 discrete action field 保持整数。若双臂都启用但只有
一侧配置末端，真实 deploy CLI 使用 `--end-effector-sides right`（或 `left`）以避免为
未配置的一侧启动 telemetry/dispatch（并按需配合 `--enable-hands`）；该选项不改变两个固定
arm pose slot。

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
- 对旧 v2.1 dataset，通用 replay 与后续模型部署应优先读取 26 维 `action` 和 action spec，
  再经同一套 `RobotAction`/`TeleopCommand` 解析路径下发。首帧 qpos init
  是机械臂定位辅助，不是模型 action schema 的一部分。
- `action.gripper.<side>.target`: DMTAC gripper 的标量夹爪目标，适合后续按 1 维 gripper 训练。
- DMTAC gripper 历史数据可能同时保留 6-slot hand 兼容向量；训练时应以
  `action.gripper.<side>.target` 作为每侧 1 维夹爪目标。其余 5 个 hand slot
  是无效 padding，可能为 `NaN`，不应仅因此判定整帧 action 损坏。
- `observation.arm.<side>.*`: 当前/实际关节、TCP、硬件回读和时序信息。
- `action.<side>_arm.pose`: 26 维根 `action` 中该侧 7D pose 的显式副本；v2.3 writer
  会验证二者一致，并按 storage contract 一起转换。
- `action.reference.<side>_arm.pose_ref`: RM65 `q_ref` 经 FK 得到的 controller/raw-base
  审计 target，不是模型 action 副本，也不参与 raw→desk 转换。v2.3 必须同时写
  `pose_ref_frame=<source.raw_absolute_frame_by_side[side]>`。
- `observation.hand.<side>.*`: 手/夹爪 telemetry、触觉索引、触觉采样状态。
  对 DMTAC，位置新鲜度只以 `joint_timestamp`（同时暴露为
  `gripper_measurement_timestamp`）判定；`telemetry_timestamp` 只是后台循环健康时间，
  跳过寄存器读取时不得刷新位置年龄。诊断列包含 `joint_age_sec`、命令开始/完成时间、
  读写耗时、串口锁等待和跳过原因。
- `observation.images.<camera_key>.*`: 视频帧索引、源时间戳、接收时间戳、
  本机历史可用时间戳和视频路径。
- `teleop.action_source.<side>_wrist_pose_desk`: 历史命名的 Vive 遥操目标 pose；它实际位于
  `vive_retarget_frame`，不是 v2.3 canonical desk。新数据必须同时写入
  `teleop.action_source.<side>_wrist_pose_desk_frame="vive_retarget_frame"`；旧数据缺少该
  frame 列时只能按这一历史语义兼容，不能推断为 canonical desk。
- `teleop.channels.*`: 操作端输入通道时间戳和有效性。
- `teleop.pose_quality.*`: 从本帧已有 IK/防碰撞 metadata 派生的 pose 质量指标；若最终
  accepted-qref 带有明确的奇异证书，则该帧会保留特殊排除标记。
- `teleop.arm_mode.*`: 从本帧 command/session metadata 派生的左右臂语义状态标签。

## v2.3 causal state snapshot contract (`dexteleop_causal_snapshot_v1`)

从 v2.3 起，落盘 state/action 的时间切点为动作信息被接收并建立的 `T0`，而不是
异步 sample builder 或 writer 实际运行的时间。手部 telemetry、触觉和 arm feedback
都先进入有界、不可变历史，再按 `T0` 查询；数组和嵌套字典在发布历史时完成深拷贝。

每个内建快照至少记录：

- `source_timestamp`：设备测量/帧产生时间；
- `available_timestamp`：完整 payload 在本机完成接收、解析、深拷贝并发布到历史后的时间；
- `source_clock`、`available_clock`、`snapshot_sequence`：时间域和稳定的选择审计信息。

严格资格为 `source_timestamp <= T0 AND available_timestamp <= T0`。没有满足条件的
快照必须以缺失/invalid 表示，不得回退到 sample build 时直接读取的可变 backend state。
手部 telemetry 与 tactile 是独立 stream：一个 row 可以选择不同的历史序号，并分别写入
`telemetry_snapshot_*` 与 `tactile_snapshot_*` 列。arm 和 wrist-camera actual pose
使用同一套 arm snapshot 选择结果。

手部的 `*_snapshot_source_age_sec` 与 `*_snapshot_available_age_sec` 分别记录测量年龄和
本机可见年龄；兼容列 `*_snapshot_age_sec` 取两者较大值。这样后台重复轮询同一份缓存
状态不会仅因重新发布而被误判为新测量。

VIVE 已完成的 clock-sync 将 sender monotonic capture time 映射到本机 wall-clock，写入
`source_timestamp`；本机 UDP 接收/发布时间仍写入 `available_timestamp`，因此“物理上早但
网络上晚”的包同样不会进入该 T0。未校准旧 sender 保留显式 receiver-wall fallback 标记。

直连 RM65、手部 telemetry 和 tactile 的 source/available clock 都是
`local_wall_time`。RealSense 不把相机固件 timestamp 当作跨设备时间：其
`source_timestamp` 是本机接收该帧的 wall time，标记 `source_clock=receiver_wall_time`；
SDK 的 `camera_timestamp_sec`、`sensor_timestamp_ms`、`timestamp_domain` 和
`sensor_frame_number` 只作设备审计。dataset/episode/row 分别固定保存 causal contract
schema 与 canonical hash，追加或验收时不允许混用不同定义。

读取时建议只依赖需要的列；调试/采集列可能随版本增加。
历史 parquet 可能没有 `teleop.pose_quality.*`，兼容读取时不应把缺失解释为坏样本。
历史 parquet 也可能没有 `teleop.arm_mode.*`；兼容读取时应按可选标签处理，
不要把缺失解释为 action/state schema 错误。

## v2.3 absolute/relative pose contract (`dexteleop_pose_frame_contract_v1`)

v2.3 的落盘 pose 仍然是绝对位姿；relative 只是训练窗口或部署 policy
输出的即时表示，不回写 parquet。这样同一份 v2.2/v2.3 数据可以按模型需要选择
absolute 或 relative，而不会产生两份相互漂移的标签。

- reader 提供给模型的 `observation.state` arm pose 永远是 canonical desktop absolute
  frame (`desk`)；若 dataset 标记为 `raw_absolute`，这是读取时转换后的视图，不代表
  parquet 原值已经位于 desk。
- `action` 与 `observation.last_action` 在存储层仍为绝对 pose；`last_action` 的
  relative 视图使用当前行的 actual TCP，而不是上一行/上一 chunk 的 anchor。
- relative action chunk 的 anchor 是该 chunk 首行、且 `<=T0` 选出的实际测量
  `observation.arm.<side>.actual_tcp_pose`；一个 chunk 内所有 action 行共用该 anchor。
- `meta/info.json`、episode metadata 或 row metadata 必须携带
  `pose_frame_contract`。其中 `source` 显式声明 `source_id`、`source_family`、
  `topology`、各 side 的 raw frame 和 raw→canonical-desk 变换；不同 side 可以来自
  不同机器人 base frame，并分别覆盖共享默认变换。该 contract 在整个 dataset 内不可变；
  `pose_storage_frame` 为
  `desk_absolute`（v2.3 默认）或强制采集的 `raw_absolute`。
- v2.2 及更早数据没有该 metadata 时，reader 按
  `PoseFrameContract.legacy_v22_raw()` 解释，即 raw RM65（按单臂/双臂 topology
  区分）且默认 raw→desk 为 identity；不会把旧数据误认成已经校准的 desk 数据。
- 训练/回放 reader 必须先把 storage frame 转换到 desk，再做 relative 解算；
  仅允许有限、合法的 SE(3) pose，不能用 command target 代替实际 TCP anchor。

当前双 RM65 station 的默认 canonical `desk` 是已标定 URDF 中的
`workspace_near_boundary_frame`：原点在桌面表面上左右居中，并且沿“桌面中心到
操作者/机械臂”的方向位于有效操作空间最靠近操作者/机械臂一侧的边界。该点在 URDF
world 下仍为 `(-0.472, 0, 0.010)`；`+X` 是操作者面向桌面的方向，`+Y` 是从该原点
朝向操作者左侧的方向，`+Z` 是垂直桌面向上的方向。
左右 controller base 到该 frame 的变换分别由 URDF fixed joint 计算为
`inv(T_world_desk) @ T_world_base[side]`；不得用历史 Vive retarget 原点或简单 identity
替代。该默认只适用于此双臂 station，单臂/其他安装必须使用独立 source profile。

这里的 canonical `desk` 与 Vive 遥操标定历史上称为 `desk` 的中间坐标系不是同一概念。
后者统一称为 `vive_retarget_frame`，旧配置键 `desk_frame`、`T_O_D`、`T_D_B` 仅作为兼容
别名；对应的新名称为 `vive_retarget_frame`、`T_O_VR`、`T_VR_B`，不得直接拿它充当
dataset raw→desk contract。

`observation.arm.<side>.actual_hardware_pose_6d` 是 RM65 SDK 原样反馈的 controller-base
`[x,y,z,rx,ry,rz]`（m/rad）审计值，使用控制器当前 TCP；它不参与 raw/desk 存储转换，
也不能作为 relative anchor。`actual_tcp_pose` 才是项目按 measured qpos、配置的运动学与
TCP 定义计算并按 pose contract 变换的标准 pose，因此两者允许存在差异。

当前 collection writer 的 v2.3 接入位于 row 构造完成、异步写任务提交之前，只处理该 row
的深拷贝，不修改 `CollectorSample`、`TeleopCommand` 或 backend state。配置
`raw_absolute` 时 pose 数值不变；配置 `desk_absolute` 时转换以下标准 arm pose 副本：

- `action`、`action.arm.pose`、`action.<side>_arm.pose`；
- `observation.state`、`observation.arm.pose`、`observation.last_action`；
- `observation.<side>_arm.pose`；
- `observation.arm.<side>.current_tcp_pose/commanded_tcp_pose/actual_tcp_pose`。

明确声明 base/controller frame 的审计字段不转换，包括
`actual_hardware_pose_6d` 和 `observation.wrist_camera.<side>.pose_base`。每行额外记录
`dexteleop.pose_storage_frame` 与 `dexteleop.pose_source_id`；dataset `info.json` 和 episode
metadata 中重复的 `pose_frame_contract` 必须完全一致，否则 writer/loader 拒绝继续。

v2.3 writer 在 pose 转换前还会逐侧验证 measured TCP：反馈必须成功且有效、未过期、
sequence 合法，并同时满足 arm snapshot 的
`source_timestamp <= T0` 与 `available_timestamp <= T0`。任一条件不满足时该 row
fail-closed，不允许用 commanded/cached pose 补写。启用 `pose_frame_contract` 时也不允许
通过 CLI 关闭 actual arm feedback。对当前 `rm65_*` source profile 还强制要求 TCP 来源为
`hardware_qpos_fk` 且保存有限的 measured hardware qpos；未来 sim/其他硬件 source 可声明
自己的 measured pose source，但同样不得使用 commanded pose。writer 同时核对每侧
`observation.<side>_arm.pose` 与该 measured `actual_tcp_pose` 一致，防止训练 state 悄然退回
commanded pose；也核对 packed `observation.state`、`observation.arm.pose` 以及所有 action
副本，避免同一帧内部出现不同坐标系或不同数值。

这组字段构成 v2.3 的 dataset contract，而不是某个模型的私有约定。转换结果应在
部署边界重新由固定的 measured TCP anchor 解回绝对 pose，并在 canonical safety/IK
之前完成。

### v2.3 training action-chunk view

固定 horizon 模型使用 `dexteleop_pose_action_chunk_v1` 显式声明训练视图。一个样本只包含
chunk 起点的一帧 observation，以及从该帧开始、长度为 `action_horizon` 的 action target：

- `observation.state` 始终是起点帧的 canonical desk absolute state；
- relative action 的整个真实前缀共用起点帧 measured `actual_tcp_pose`；
- `observation.last_action` 使用起点帧当前 measured TCP 作为 anchor；
- window 不能跨 episode；默认 `tail_policy=drop`，只暴露完整窗口；模型可在自己的
  `PoseActionChunkContract` 中显式选择 `pad_last`，使每个真实帧都可成为起点；
- `pad_last` 在 raw/desk 和 absolute/relative 转换全部完成后重复最后一个有效 action，
  并输出连续尾部 mask `action_is_pad`；不虚构 observation、state 或 TCP anchor；
- normalization statistics 和 action loss 必须排除 `action_is_pad=true` 的位置。

`dataset_interface.pose_chunk` 提供确定性的 episode/start→sample 转换和合法 start 索引；
随机采样、batch 组织及具体 learner 仍由模型训练实现负责。chunk contract 的 canonical hash
必须随 v2.3 policy artifact 保存，部署据其 `action_pose_mode` 绑定 absolute/relative 解码。
artifact 还必须用 `pose_training_data_contract` 区分 `native_v23_dataset` 与
`v22_raw_identity_compatibility`：后者只是把旧 raw state 数值按 identity compatibility 暴露给
模型，不能在部署时改用新的非 identity canonical-desk 变换；具体启动约束见
`DEPLOY_SCHEMA.md`。

## 外置媒体

视频：

- 路径来自 `episodes.jsonl.video_paths`。
- parquet 中 `observation.images.<camera_key>.video_frame_index` 指向 mp4 帧号。
- 相机样本按采集时的 action timestamp 对齐；新数据应使用 `latest <= action_timestamp`，避免读取未来帧。

触觉：

- 路径来自 `episodes.jsonl.tactile_paths`。
- `manifest.json` 描述 `format`, `frame_count`, `sensor_keys`, 每个 sensor 的 binary stream 路径、dtype、shape。
- 新采集默认对外置触觉流不压缩，按 C-order 写入裸 `.bin`；也可以用 `--tactile-compression zstd|lz4|none` 选择无损 chunk 压缩。选择 `none` 时保持旧式裸流格式。
- 压缩流在每个 stream spec 中记录 `encoding`, `compression`, `chunk_frame_count`, `chunks`。`chunks` 包含 `frame_start`, `frame_count`, `offset`, `compressed_bytes`, `raw_bytes`，用于随机读取单个 chunk。旧裸流可没有这些字段，reader 应按 `raw_c_order`/`none` 兼容。
- 正式后处理的 canonical 触觉格式为 `dexteleop_tactile_compaction_v1`。raw profile 固定为
  `dmtac_raw_120x160_q6_delta_zstd_v1`：不改变触觉帧数、时间戳或采样频率，把
  `uint8[480,640]` 确定性降为 `uint8[120,160]`，6-bit 中点重建后使用独立 chunk 的
  `chunked_delta_zstd`。force profile 固定为 `dmtac_force_qint16_zstd_v1`：逻辑值仍为
  `float32[288,384,3]`，物理存储为 chunk/channel scale 对应的 int16 Zstd。
- compact stream 的 `shape/dtype` 是 reader 返回的逻辑数组契约；`storage_dtype` 是物理值类型；
  `tactile_compaction.source` 保存迁移前路径、shape、dtype、encoding 和 logical SHA-256。
  chunked force 的 scale 保存在各 chunk 的 `quantization` 中。所有读取端必须通过
  `dataset_contracts.ChunkedTactileStream` 解码，禁止直接按文件后缀推测格式。
- Parquet 的 `tactile_raw_shape_per_sensor` 与历史 `tactile_shapes_json` 作为采集源审计字段保留；
  compact 后实际数组 shape 以 manifest stream 为准。manifest frame record 的 `shapes/dtypes`
  更新为逻辑输出，并用 `source_shapes/source_dtypes` 保存采集源值。
- parquet 中 `observation.hand.<side>.tactile_frame_index` 指向 manifest 中的触觉帧。
- 同一个不可变 `tactile_snapshot_sequence` 被多个 dataset row 选中时只写一次大数组；
  这些 row 可以合法共享同一个 `tactile_frame_index`，manifest 保留 snapshot sequence 和
  source/available clock 以供审计。
- 触觉可能相对视觉/action 有延迟，reader 应使用 manifest 与 parquet 中的 timestamp/索引做一致性检查。

## v2.3 finalized dataset acceptance

正式采集结束后运行只读验收：

```bash
python -m utils.dataset_postprocess validate-v23 /path/to/dataset
```

退出码 `0` 表示 metadata/hash、pose frame binding、measured TCP/state/action 副本、所有
`<=T0` causal cut、episode/parquet 计数和外置 artifact 引用均通过；退出码 `1` 表示至少一项
失败。该命令不会 padding、转换或改写数据，旧 v2.2 数据也不会被冒充成 native v2.3。

腕部深度 mask：

- 路径来自 `episodes.jsonl.wrist_depth_mask_paths`。
- `manifest.json` 描述 bitpack stream、图像 shape、每帧时间戳和有效深度比例。
- `stream.bitpack.bin` 使用 `np.packbits`，语义是指定深度范围内为 1，非法或范围外为 0。

腕部 final keep-mask：

- 推荐由 `utils.dataset_postprocess.wrist_cam_adjust compose-final-mask` 生成。
- 默认语义是 `final_keep_mask = depth_range_mask_postprocessed OR table_geometry_keep_mask`。
- final 合成阶段不额外添加 proxy/body mask；如果上游 table keep-mask 已包含 painted
  body mask，final keep-mask 会按输入保留。
- `manifest.json` 使用与 wrist depth mask 兼容的 bitpack stream schema；`mask_family`
  可为 `wrist_final_keep`。
- `value_1` 表示训练和增强时应保留原始 RGB 的像素；`value_0` 表示可替换背景。

## 训练读取约定

- `episode_index` 是 episode 身份编号，不是数组下标；删除/抽取后可能不连续。若后处理把一个 episode 切成多个 segment，可重新生成唯一的 `episode_index`，并在 metadata 中保留 `source_episode_index`。
- `index` 是当前 dataset 的扁平 row id；复制式后处理产物应重建 `index`，保证当前输出 dataset 内唯一且连续。需要追溯原始编号时，应使用派生 metadata 或额外 source 字段，而不是复用 `index`。
- v2.2 的 `policy_io_layout` / `policy_io_layout_hash` 是维度和语义权威来源；旧数据才以
  `features` 和 action spec 为准。下游不得硬编码 `action=26` 或 `observation.state=38`。
- 对 DMTAC gripper，新数据应按每侧 1 维 gripper target 训练；历史数据如果仍保留 6 slot hand target，需要通过 `action.gripper.<side>.target`、action spec 或后处理转换到训练所需格式。每侧未使用的 5 个 hand slot 是 padding，不应当作训练 action 维度。
- replay、部署和模型推理应共享同一个 action contract：v2.2 先校验 layout hash，然后 arm 使用 7D pose，
  hand/gripper 使用 `hand_action_specs_by_side` 声明的 active slots。末端类型变化时，
  优先新增或更新 action spec 与后端适配器，不应在 replay、训练 reader 或 deploy
  入口分别硬编码维度。
- `FAULT/HOLD` 后继续写入的尾段、长时间卡顿分段、以及人工标记的碰撞 episode，应通过 `utils.dataset_postprocess` 生成新的 clean/collision dataset，而不是原地改写原始数据。
- 后处理产物应更新 `meta/info.json`, `episodes.jsonl`, `episodes_stats.jsonl`，并保留 `postprocess_history`。
- 通用 masked image/video 增强应通过 `dataset_interface` 接入下游模型。该层只负责读取
  RGB 视频、读取 final keep-mask、按 mask 在线增强并返回 NumPy/Tensor 友好数组；
  不负责生成 mask、不修改 dataset、不定义模型归一化策略。
- `dataset_interface` 默认把 `mask > 0` 解释为 keep region。DataLoader worker 内应各自持有
  video reader cache、bitpack mask reader cache 和 background stream cache，避免每帧重新打开
  mp4 或 mask 文件。
- `episode_index` 仍按身份编号解析，不应当作连续数组下标；`dataset_interface` 应从
  `episodes.jsonl` 建索引后解析视频和 mask 路径。

## Arm Mode 特殊状态标签

`arm_mode` 是逐帧、逐侧机械臂语义状态标注，用于让模型区分“正常遥操”、
“人为固定/保持不动”与“执行 init pose”。它是旁路标签，不参与 action
拼接，不改变 `action`, `observation.state`, `observation.last_action` 的形状或语义。

完整 payload 位于 `sample_metadata_json.arm_mode`，当前 schema 名为
`dexteleop_arm_mode_v2`（读取端仍须兼容 v1）。Parquet 中同时展开以下模型 mode 列：

- `teleop.arm_mode.schema`: 当前为 `dexteleop_arm_mode_v2`（兼容读取 v1）。
- `teleop.arm_mode.available`: `0/1`，表示本帧是否有 arm mode metadata。
- `teleop.arm_mode.<side>.id`: 训练友好的整数枚举。
- `teleop.arm_mode.<side>.label`: 字符串枚举。
- `teleop.arm_mode.<side>.is_special`: `0/1`，`fixed_hold` 和 `init_pose` 为特殊状态。
- `teleop.arm_mode.<side>.source`: 标签来源，例如 `normal`, `fix_arm`, `init_arm`,
  `command_hold`, `recording_pause`, `session_state`, `inactive_side`。
- `teleop.arm_mode.<side>.tags`: 可扩展标签列表。当前可能包含 `fix_arm`, `init_arm`,
  `home_recapture`。
- `teleop.arm_mode_json`: 完整 JSON，包含每侧 `mode/id/source/tags/detail`。

另有仅供审计的列：

- `teleop.arm_mode.<side>.scope`: fix 的实际覆盖范围，
  `arm_and_end_effector` 或 `arm_only`。
- `teleop.arm_mode.<side>.init_phase`: `not_active`, `init_move`, `init_hold`, `unknown`。
- `teleop.arm_mode.<side>.init_phase_valid`、目标误差、实际 TCP 线/角速度及 dwell 时间。

上述 `init_phase`/scope 字段不进入 dataset interface、训练 batch、模型 token/embedding、
loss 或部署 observation。模型对 init 只有 `init_pose` 一个标签。

当前枚举：

- `unknown = 0`: 缺失或无法判断。
- `teleop = 1`: 正常遥操。
- `fixed_hold = 2`: fix-arm 固定侧保持上一帧下发目标动作。该标签不是硬件急停。
- `init_pose = 3`: init-arm 侧正在前往或保持 init pose；模型侧不区分移动和静止。
- `stopped = 4`: 非 RUNNING session、已写入的停止路径，或精确 qpos Init 被第二次
  同侧踏板暂停后等待 tracker Home 重采的冻结帧。

判定优先级：

1. 如果本帧处于 recording pause 或非 `RUNNING` session，左右活动臂标为 `stopped`。
2. 否则如果 command 明确声明精确 qpos Init 的目标臂因 Home 重采而冻结，
   仅目标臂标为 `stopped`；该侧不再带 `init_arm` 标签，非目标臂保持
   `teleop` 或其自身独立状态。
3. 否则该侧处于 init-arm active（包括旧 pose Init 的 recapturing），标为 `init_pose`。
4. 否则该侧处于 fix-arm fixed/recapturing，标为 `fixed_hold`。
5. 否则活动侧标为 `teleop`，非活动侧标为 `unknown`。

四踏板精确 qpos Init 允许在已开始的正式 episode 内触发，以便这些逐帧标签实际进入
Parquet；episode 开始倒计时、后台保存以及另一个精确 qpos operation 活动期间仍拒绝新请求。
第二次踩同侧 Init 踏板是标签边界：后续 Home 重采帧从 `init_pose` 切换为 `stopped`，
Home 完成后的下一条普通遥操 command 才恢复为 `teleop`。

单侧精确 Init 成功后保持 `init_pose`/Init-fixed，非目标臂继续遥操，
不会自动触发全局 Hold。只有左右两侧均被 `fixed_hold` 或 Init-fixed
覆盖时，运行时进入可自动解除的 Fix-Hold 并暂停当前 episode；任一侧
Home 重采完成后经 warmup 继续同一 episode。Fix-Hold 期间按普通 Hold
会升级为常规 Hold，清除特殊状态并要求双臂 Home。Fix-Hold 暂停期间不写
Parquet row，因此不会产生“双臂静止”的无效训练窗。

GUI 的“双臂顺序精确归位”是独立的维护流程：只按左臂→右臂顺序
执行 qpos 路径，中间和完成时都不重采 tracker Home。左臂实测到位后
释放该侧 exact-qpos 控制权再启动右臂；右臂到位后立即进入常规
Hold。任一步失败也进入常规 Hold，不再执行后续臂动作。因为常规 Hold
会暂停已打开的 episode，该流程不会在旧 Home 与归位后机器人位姿之间
自动建立新的遥操映射。操作员必须在 Hold 中显式重采 runtime Home 才能恢复。

Fix/Init 的单侧 Home 重采只接受有效、时龄不超过 `0.5 s` 且 source
timestamp 严格递增的 tracker 帧。tracker 需连续静止至少 `0.25 s`，位移/
旋转速度上限分别为 `0.015 m/s` 和 `2.5 deg/s`；采样窗中姿态离散超过
`3 mm` 或 `0.75 deg` 时清空窗口重新计时。这些约束防止重复旧帧或运动中
tracker 被写入 Home。如果部分 Home anchor 已应用后后续步骤失败，运行时
fail closed 进入常规 Hold。

正式数采中触发 hold 时，当前 episode 保持打开但暂停写样本；同时 fix/init
特殊状态会停止。`task_hold` 只用于运行时/episode 审计，不生成数据集标签。再次
hold/re-home 后恢复普通遥操，后续写入帧应从 `teleop` 重新开始，而不是继续继承 hold
前的 `fixed_hold` 或 `init_pose`。

`init_phase=init_hold` 只在实际、因果有效的 TCP 反馈相对 init target 满足 1.5 mm /
0.25° 误差、低于静息微动速度阈值并连续 3 s 后写入。缺少真实反馈、target 或时间连续性
时写 `unknown` 且 `init_phase_valid=0`，绝不由下发 command 推断。

`fix_arm` 必须锁存该侧全部已配置 effector（arm 与 hand/gripper）。若某侧没有配置
末端执行器，允许 `arm_only`，但完整 JSON 的 `detail.fix_arm_warning` 会明确说明该退化。

兼容旧数据时，如果缺少 `teleop.arm_mode.*`，reader 应允许加载。需要补齐统一字段时，
推荐策略是：`session_state == RUNNING` 的活动侧按 `teleop` 填充；无法确认活动侧或
非 RUNNING 行按 `unknown` 或 `stopped` 填充，并在派生 metadata 中记录这是 legacy/default。

## Pose Quality 与碰撞标签

`pose_quality` 是逐帧 pose 质量标注；力/力矩超限碰撞只做 episode 级轨迹标签，用于后续筛选、统计或构造负样本。它们都是采集记录端派生信息，不参与 IK 求解、双臂防碰撞候选选择、轨迹限幅或真实下发逻辑。当前实现不会额外求 IK、不会额外跑 collision query，也不会新增力传感器读取；所有判断都来自该帧已经存在的 command metadata、faults 和硬件诊断文本。

兼容旧数据时，本节所有质量标注都应按可选字段处理。如果 `pose_quality` 或 `trajectory_quality` 不存在，默认不因为缺失而降权或判坏；需要补齐统一字段时，可按正常样本填充 `teleop.pose_quality.score=1.0`, `teleop.pose_quality.status=good`, `teleop.pose_quality.special_code=null`, `teleop.pose_quality.bad=0`, `teleop.pose_quality.available=0`，并在派生 metadata 中注明这是 legacy/default 兼容值。

### 逐帧 `pose_quality`

完整 payload 位于 `sample_metadata_json.pose_quality`，schema 名为
`dexteleop_pose_quality_v1`。parquet 中同时展开以下常用列：

- `teleop.pose_quality.score`: 普通帧为 `0.0` 到 `1.0` 的启发式质量分。`1.0` 表示当前已记录信号下未发现风险；`0.0` 表示风险项较多或较重。明确进入奇异 qref 时固定写 `-1.0`，该值是排除标记，不参与 episode 的普通分数均值/最小值。
- `teleop.pose_quality.status`: `good`, `warning`, `bad`, `singularity`, `unknown` 之一。
- `teleop.pose_quality.special_code`: 特殊质量代码；当前 `singularity` 表示接受的 qref 带有奇异/奇异逃逸证书。普通帧为 `null`。
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

奇异特殊标记只读取已存在的 accepted-qref metadata，不在采集端重新求 IK：
`ik_candidate_selection.candidate_certificate` 中的
`singular_fallback_claimed`、`candidate_selected_singular_fallback_claimed`、
`singularity_aware_*_traversal_by_side` 和
`final_qref_*_singularity`，以及兼容的 `singularity_event`/singularity preference
字段。普通 SDK advisory（未被最终 qref 证书接受）不会单独触发 `-1`。

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
- `pose_quality_special_count`, `pose_quality_singularity_count`: 特殊质量帧及其中奇异帧的数量。
- `pose_quality_valid_score_count`: 参与普通分数统计的非负有限分数帧数。
- `pose_quality_min_score`, `pose_quality_mean_score`: 本 episode 的 pose quality 最小/平均分。
- `force_collision_detected`: episode 内是否出现过力/力矩 hard-stop 超限。
- `force_collision_label`: 若出现过超限则为 `detected_collision`，否则为 `clean`。

为方便快速筛选，`episodes.jsonl.summary` 顶层也会写入
`force_collision_detected` 和 `force_collision_label`。`episodes_stats.jsonl`
同步写入同名统计字段。
